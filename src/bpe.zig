//! Byte-level BPE, as GPT-2, Llama 3 and Qwen use it, from the vocabulary and merge list
//! a GGUF file carries. Input bytes map to the byte-level alphabet (printable bytes to
//! themselves, the rest to code points from U+0100), text is pre-split into words much
//! like the model's regex, each word starts as one symbol per character and merges are
//! applied by rank until none apply. Decoding reverses the alphabet.
const std = @import("std");
const gguf = @import("gguf.zig");

/// Map from byte to the code point the alphabet uses for it.
fn byteToCp(b: u8) u21 {
    if ((b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE)) return b;
    // the rest are assigned 256, 257, ... in increasing byte order
    var n: u21 = 256;
    var i: u16 = 0;
    while (i < b) : (i += 1) {
        const c: u8 = @intCast(i);
        if (!((c >= '!' and c <= '~') or (c >= 0xA1 and c <= 0xAC) or (c >= 0xAE))) n += 1;
    }
    return n;
}

fn cpToByte(cp: u21) ?u8 {
    if (cp < 256) {
        const b: u8 = @intCast(cp);
        if ((b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE)) return b;
        return null;
    }
    var n: u21 = 256;
    var i: u16 = 0;
    while (i < 256) : (i += 1) {
        const c: u8 = @intCast(i);
        if (!((c >= '!' and c <= '~') or (c >= 0xA1 and c <= 0xAC) or (c >= 0xAE))) {
            if (n == cp) return c;
            n += 1;
        }
    }
    return null;
}

pub const Tokenizer = struct {
    gpa: std.mem.Allocator,
    /// id -> token text in alphabet form (as stored in the file).
    tokens: [][]const u8,
    /// alphabet-form text -> id
    ids: std.StringHashMapUnmanaged(u32),
    /// "left right" -> rank
    merges: std.StringHashMapUnmanaged(u32),
    /// id -> decoded bytes (owned)
    decoded: [][]u8,
    eos: u32,
    bos: ?u32,
    im_start: ?u32,
    im_end: ?u32,
    /// Byte->alphabet table, as UTF-8 strings.
    alpha: [256][]const u8,
    alpha_buf: [256 * 2]u8,

    pub fn init(gpa: std.mem.Allocator, f: *const gguf.File) !*Tokenizer {
        const t = try gpa.create(Tokenizer);
        errdefer gpa.destroy(t);
        t.gpa = gpa;
        const toks = f.getArray("tokenizer.ggml.tokens") orelse return error.NoVocab;
        t.tokens = try gpa.alloc([]const u8, toks.count);
        errdefer gpa.free(t.tokens);
        t.ids = .empty;
        errdefer t.ids.deinit(gpa);
        try t.ids.ensureTotalCapacity(gpa, @intCast(toks.count));
        var it = toks.strings();
        var i: usize = 0;
        while (it.next()) |s| : (i += 1) {
            t.tokens[i] = s;
            t.ids.putAssumeCapacity(s, @intCast(i));
        }
        t.merges = .empty;
        errdefer t.merges.deinit(gpa);
        if (f.getArray("tokenizer.ggml.merges")) |m| {
            try t.merges.ensureTotalCapacity(gpa, @intCast(m.count));
            var mi = m.strings();
            var rank: u32 = 0;
            while (mi.next()) |s| : (rank += 1) t.merges.putAssumeCapacity(s, rank);
        }
        // alphabet table
        var pos: usize = 0;
        for (0..256) |b| {
            const n = std.unicode.utf8Encode(byteToCp(@intCast(b)), t.alpha_buf[pos..]) catch unreachable;
            t.alpha[b] = t.alpha_buf[pos .. pos + n];
            pos += n;
        }
        // decoded forms
        t.decoded = try gpa.alloc([]u8, toks.count);
        errdefer gpa.free(t.decoded);
        for (t.tokens, 0..) |s, id| {
            var out: std.ArrayList(u8) = .empty;
            var view = std.unicode.Utf8View.initUnchecked(s);
            var cit = view.iterator();
            while (cit.nextCodepoint()) |cp| {
                if (cpToByte(cp)) |b| try out.append(gpa, b) else {
                    var buf: [4]u8 = undefined;
                    const n = std.unicode.utf8Encode(cp, &buf) catch 0;
                    try out.appendSlice(gpa, buf[0..n]);
                }
            }
            t.decoded[id] = try out.toOwnedSlice(gpa);
        }
        t.eos = f.getU32("tokenizer.ggml.eos_token_id") orelse 0;
        t.bos = f.getU32("tokenizer.ggml.bos_token_id");
        t.im_start = t.ids.get("<|im_start|>");
        t.im_end = t.ids.get("<|im_end|>");
        return t;
    }

    pub fn deinit(t: *Tokenizer) void {
        for (t.decoded) |d| t.gpa.free(d);
        t.gpa.free(t.decoded);
        t.merges.deinit(t.gpa);
        t.ids.deinit(t.gpa);
        t.gpa.free(t.tokens);
        t.gpa.destroy(t);
    }

    pub fn text(t: *const Tokenizer, id: u32) []const u8 {
        return if (id < t.decoded.len) t.decoded[id] else "";
    }

    fn isLetter(c: u8) bool {
        return std.ascii.isAlphabetic(c) or c >= 0x80;
    }

    /// Split like the GPT-2/Qwen pattern: contractions, letters with an optional leading
    /// non-letter, single digits, punctuation runs with an optional space, whitespace.
    fn preTokenize(gpa: std.mem.Allocator, s: []const u8, out: *std.ArrayList([]const u8)) !void {
        var i: usize = 0;
        const contractions = [_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D" };
        while (i < s.len) {
            var hit = false;
            for (contractions) |c| if (std.mem.startsWith(u8, s[i..], c)) {
                try out.append(gpa, s[i..][0..c.len]);
                i += c.len;
                hit = true;
                break;
            };
            if (hit) continue;
            const start = i;
            var j = i;
            // optional single non-letter non-digit non-newline prefix before letters
            if (!isLetter(s[j]) and !std.ascii.isDigit(s[j]) and s[j] != '\r' and s[j] != '\n' and j + 1 < s.len and isLetter(s[j + 1])) j += 1;
            if (j < s.len and isLetter(s[j])) {
                while (j < s.len and isLetter(s[j])) j += 1;
                try out.append(gpa, s[start..j]);
                i = j;
                continue;
            }
            j = start;
            if (std.ascii.isDigit(s[j])) {
                try out.append(gpa, s[j .. j + 1]);
                i = j + 1;
                continue;
            }
            // punctuation run, optional leading space, trailing newlines
            if (s[j] == ' ' and j + 1 < s.len and !std.ascii.isWhitespace(s[j + 1]) and !isLetter(s[j + 1]) and !std.ascii.isDigit(s[j + 1])) j += 1;
            if (j < s.len and !std.ascii.isWhitespace(s[j]) and !isLetter(s[j]) and !std.ascii.isDigit(s[j])) {
                while (j < s.len and !std.ascii.isWhitespace(s[j]) and !isLetter(s[j]) and !std.ascii.isDigit(s[j])) j += 1;
                while (j < s.len and (s[j] == '\r' or s[j] == '\n')) j += 1;
                try out.append(gpa, s[start..j]);
                i = j;
                continue;
            }
            // whitespace: newlines together; other spaces keep one back for the next word
            j = start;
            if (s[j] == '\r' or s[j] == '\n') {
                while (j < s.len and (s[j] == '\r' or s[j] == '\n')) j += 1;
                try out.append(gpa, s[start..j]);
                i = j;
                continue;
            }
            while (j < s.len and std.ascii.isWhitespace(s[j]) and s[j] != '\r' and s[j] != '\n') j += 1;
            if (j < s.len and !std.ascii.isWhitespace(s[j]) and j - start > 1) j -= 1;
            if (j == start) j += 1;
            try out.append(gpa, s[start..j]);
            i = j;
        }
    }

    /// Encode text (no special tokens). Caller owns the result.
    pub fn encode(t: *const Tokenizer, gpa: std.mem.Allocator, s: []const u8) ![]u32 {
        var words: std.ArrayList([]const u8) = .empty;
        defer words.deinit(gpa);
        try preTokenize(gpa, s, &words);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        var syms: std.ArrayList([]const u8) = .empty;
        defer syms.deinit(gpa);
        var pair: std.ArrayList(u8) = .empty;
        defer pair.deinit(gpa);
        var merged_store: std.ArrayList(u8) = .empty; // arena for merged symbol text
        defer merged_store.deinit(gpa);
        for (words.items) |w| {
            syms.clearRetainingCapacity();
            merged_store.clearRetainingCapacity();
            var alpha_len: usize = 0;
            for (w) |b| {
                try syms.append(gpa, t.alpha[b]);
                alpha_len += t.alpha[b].len;
            }
            // every merge appends at most the whole word; reserve so slices stay valid
            try merged_store.ensureTotalCapacity(gpa, alpha_len * (w.len + 1));
            // merge by lowest rank until none
            while (syms.items.len > 1) {
                var best_rank: u32 = std.math.maxInt(u32);
                var best: usize = 0;
                for (0..syms.items.len - 1) |k| {
                    pair.clearRetainingCapacity();
                    try pair.appendSlice(gpa, syms.items[k]);
                    try pair.append(gpa, ' ');
                    try pair.appendSlice(gpa, syms.items[k + 1]);
                    if (t.merges.get(pair.items)) |r| if (r < best_rank) {
                        best_rank = r;
                        best = k;
                    };
                }
                if (best_rank == std.math.maxInt(u32)) break;
                // the merged text has to outlive this loop; keep it in the store and
                // re-slice since the store may move
                const a = syms.items[best];
                const b = syms.items[best + 1];
                const base = merged_store.items.len;
                merged_store.appendSliceAssumeCapacity(a);
                merged_store.appendSliceAssumeCapacity(b);
                syms.items[best] = merged_store.items[base .. base + a.len + b.len];
                _ = syms.orderedRemove(best + 1);
            }
            for (syms.items) |sy| {
                if (t.ids.get(sy)) |id| try out.append(gpa, id) else {
                    // unknown symbol: fall back to its bytes one at a time
                    var view = std.unicode.Utf8View.initUnchecked(sy);
                    var cit = view.iterator();
                    while (cit.nextCodepointSlice()) |cp| if (t.ids.get(cp)) |id| try out.append(gpa, id);
                }
            }
        }
        return out.toOwnedSlice(gpa);
    }
};

test "byte alphabet round trips" {
    for (0..256) |b| try std.testing.expectEqual(@as(?u8, @intCast(b)), cpToByte(byteToCp(@intCast(b))));
    try std.testing.expectEqual(@as(u21, 'A'), byteToCp('A'));
    try std.testing.expectEqual(@as(u21, 0x120), byteToCp(' ')); // GPT-2's "Ġ"
}

test "pre-tokenization" {
    const gpa = std.testing.allocator;
    var words: std.ArrayList([]const u8) = .empty;
    defer words.deinit(gpa);
    try Tokenizer.preTokenize(gpa, "Carl's 42 dogs, \"run\"!\n", &words);
    const want = [_][]const u8{ "Carl", "'s", " ", "4", "2", " dogs", ",", " \"", "run", "\"!\n" };
    try std.testing.expectEqual(want.len, words.items.len);
    for (want, words.items) |w, g| try std.testing.expectEqualStrings(w, g);
}
