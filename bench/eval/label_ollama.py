#!/usr/bin/env python3
"""Label transcript windows with the map worker's model through Ollama, with the mood
constrained to zwhisper's vocabulary. usage: label_ollama.py out.json name=transcript.jsonl ..."""
import json, sys, urllib.request, os, re, time
URL = os.environ.get("OLLAMA_URL", "http://10.0.0.11:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:30b-a3b")
MOODS = ["warm solo piano, slow, tender", "driving percussion, fast, urgent", "tense low strings, slow pulse, ominous",
         "soft ambient pad, still, curious", "bright acoustic guitar, light, hopeful", "dark drones, sparse, dread",
         "playful pizzicato, quick, mischievous", "epic brass and drums, big, triumphant", "gentle harp and flute, calm, wonder",
         "pulsing synth, steady, focused", "melancholy cello, slow, grief", "shimmering strings, rising, awe",
         "quiet guitar, unhurried, nostalgic", "rhythmic marching drums, determined", "eerie high strings, uneasy, mystery",
         "lively folk fiddle, upbeat, festive"]
PROMPT = """You are scoring an audiobook with background music, like a film composer.
Below is a few minutes of a book, as timed lines. Times are seconds from the start of this part.
Split it into scenes, in order. A scene changes when the mood, place, or action changes; a few minutes usually hold 1 to 4 scenes.
For each scene give:
- start: the time of the line where it begins (the first scene starts at 0)
- summary: one short sentence of what happens
- music: "score" for background music, "silence" for moments that need none (a quiet, intimate, or emotionally raw exchange, a pause for effect), or "song" only when the text says a specific real song is playing
- mood: for "score", the music to play, chosen from this list exactly: {moods}
- intensity: 1 (barely there) to 5 (full action). Conversation and explanation are 1 or 2; danger or action is 4 or 5.
- song_artist and song_title: only for "song", the song the text names
Lines:
{lines}"""
SCHEMA = {"type": "object", "properties": {"scenes": {"type": "array", "items": {"type": "object", "properties": {
    "start": {"type": "number"}, "summary": {"type": "string"},
    "music": {"type": "string", "enum": ["score", "silence", "song"]},
    "mood": {"type": "string", "enum": MOODS + [""]},
    "intensity": {"type": "integer", "minimum": 1, "maximum": 5},
    "song_artist": {"type": "string"}, "song_title": {"type": "string"}},
    "required": ["start", "summary", "music", "mood", "intensity"]}}}, "required": ["scenes"]}

def lines_of(path):
    out = []
    for l in open(path):
        if not l.strip(): continue
        s = json.loads(l); t = s["t0"] // 100
        out.append(f"[{t//60}:{t%60:02d} = {t}] {s['text'].strip()}")
    return "\n".join(out)

def ask(lines):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(moods=json.dumps(MOODS), lines=lines)}],
                       "format": SCHEMA, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_ctx": 16384}}).encode()
    req = urllib.request.Request(URL + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(json.loads(r.read())["message"]["content"]).get("scenes") or []

out_path, items = sys.argv[1], sys.argv[2:]
labels = json.load(open(out_path)) if os.path.exists(out_path) else {}
for it in items:
    name, path = it.split("=", 1)
    if name in labels: continue
    t = time.time()
    scenes = ask(lines_of(path))
    scenes.sort(key=lambda s: s["start"])
    if scenes: scenes[0]["start"] = 0
    for s in scenes:  # "song" without a named song is just score
        if s["music"] == "song" and not (s.get("song_title") or "").strip():
            s["music"] = "score"
            if s.get("mood", "") not in MOODS: s["mood"] = "soft ambient pad, still, curious"
    labels[name] = [{"start": int(s["start"]), "music": s["music"], "mood": s.get("mood", "") if s["music"] == "score" else "",
                     "intensity": int(s["intensity"]), "why": s.get("summary", "")} for s in scenes]
    json.dump(labels, open(out_path, "w"), indent=1)
    print(f"{name}: {len(scenes)} scenes in {time.time()-t:.0f} s", flush=True)
