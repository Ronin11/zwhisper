#!/usr/bin/env python3
"""Label chunks on the three axes with an Ollama model. Resumable; one JSON line per chunk.
usage: label.py <model> [out.jsonl]   env OLLAMA_URL"""
import json, sys, os, urllib.request, time, re
URL = os.environ.get("OLLAMA_URL", "http://10.0.0.11:11434")
MODEL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"labels_{MODEL.replace(':', '_').replace('.', '')}.jsonl"
POLICY = open("policy.md").read()
SCHEMA = {"type": "object", "properties": {"silence": {"type": "boolean"}, "energy": {"type": "integer", "minimum": 1, "maximum": 5},
          "valence": {"type": "string", "enum": ["dark", "neutral", "light"]}}, "required": ["silence", "energy", "valence"]}
chunks = [json.loads(l) for l in open("chunks.jsonl")]
prev = {}
for c in chunks: prev[c["id"]] = next((p for p in chunks if p["window"] == c["window"] and p["idx"] == c["idx"] - 1), None)
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT) if l.strip()}
out = open(OUT, "a")
t_start = time.time(); k = 0
for c in chunks:
    if c["id"] in done: continue
    p = prev[c["id"]]
    user = (f"EARLIER PASSAGE (context only, do not rate):\n{p['text']}\n\n" if p else "") + f"PASSAGE TO RATE:\n{c['text']}\n\nAnswer as JSON with keys silence, energy, valence."
    body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": POLICY}, {"role": "user", "content": user}],
                       "format": SCHEMA, "stream": False, "think": False, "options": {"temperature": 0, "num_ctx": 4096}}).encode()
    req = urllib.request.Request(URL + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = json.load(r)["message"]["content"]
                try: ans = json.loads(raw)
                except ValueError:
                    m = {k: re.search(k + r'\W+(\w+)', raw, re.I) for k in ("silence", "energy", "valence")}
                    if not all(m.values()): raise ValueError("unparsable: " + raw[:80])
                    ans = {"silence": m["silence"].group(1).lower() == "true", "energy": int(m["energy"].group(1)), "valence": m["valence"].group(1).lower()}
                    if ans["valence"] not in ("dark", "neutral", "light"): raise ValueError("bad valence: " + raw[:80])
                break
        except Exception as e:
            print("retry", c["id"], e, file=sys.stderr); time.sleep(2); ans = None
    if ans is None: continue
    ans["id"] = c["id"]; ans["energy"] = max(1, min(5, int(ans["energy"])))
    out.write(json.dumps(ans) + "\n"); out.flush(); k += 1
    if k % 50 == 0: print(f"{k} labeled, {(time.time() - t_start) / k:.2f} s each", file=sys.stderr)
print("done", k, file=sys.stderr)
