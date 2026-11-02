//! SentencePiece-style BPE from a Hugging Face tokenizer.json, as Gemma and the bekko
//! encoders use it: spaces become "▁" (Metaspace, one prepended), text splits before
//! each "▁", every piece starts as one symbol per character (or its bytes, through the
//! <0xNN> fallback tokens), and merges apply by rank, leftmost first on ties.
const std = @import("std");

pub const metaspace = "\xe2\x96\x81";

const Merge = packed struct { rank: u32, id: u32 };

pub const Tokenizer = struct {
    gpa: std.mem.Allocator,
    parsed: std.json.Parsed(std.json.Value),
    /// id -> piece text (as stored; "▁" for a space)
    tokens: [][]const u8,
    ids: std.StringHashMapUnmanaged(u32),
    /// (left id << 32 | right id) -> merge
    merges: std.AutoHashMapUnmanaged(u64, Merge),
    /// byte -> <0xNN> token id, or null when absent from the vocabulary
    bytes: [256]?u32,
    added: []const Added,
    bos: u32,
    eos: u32,
    unk: u32,

    pub const Added = struct { text: []const u8, id: u32 };

    pub fn init(gpa: std.mem.Allocator, json_bytes: []const u8) !*Tokenizer {
        const t = try gpa.create(Tokenizer);
        errdefer gpa.destroy(t);
        t.* = undefined;
        t.gpa = gpa;
        t.parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_bytes, .{});
        errdefer t.parsed.deinit();
        const root = t.parsed.value.object;
        const model = (root.get("model") orelse return error.BadTokenizer).object;
        const vocab = (model.get("vocab") orelse return error.BadTokenizer).object;
        var n_vocab: usize = 0;
        var vit = vocab.iterator();
        while (vit.next()) |e| n_vocab = @max(n_vocab, @as(usize, @intCast(e.value_ptr.integer)) + 1);
        // added tokens may sit past the vocab
        var added_list: std.ArrayList(Added) = .empty;
        errdefer added_list.deinit(gpa);
        if (root.get("added_tokens")) |al| for (al.array.items) |a| {
            const id: usize = @intCast(a.object.get("id").?.integer);
            n_vocab = @max(n_vocab, id + 1);
            try added_list.append(gpa, .{ .text = a.object.get("content").?.string, .id = @intCast(id) });
        };
        t.added = try added_list.toOwnedSlice(gpa);
        errdefer gpa.free(t.added);
        t.tokens = try gpa.alloc([]const u8, n_vocab);
        errdefer gpa.free(t.tokens);
        @memset(t.tokens, "");
        t.ids = .empty;
        errdefer t.ids.deinit(gpa);
        try t.ids.ensureTotalCapacity(gpa, @intCast(n_vocab));
        vit = vocab.iterator();
        while (vit.next()) |e| {
            const id: u32 = @intCast(e.value_ptr.integer);
            t.tokens[id] = e.key_ptr.*;
            t.ids.putAssumeCapacity(e.key_ptr.*, id);
        }
        for (t.added) |a| {
            t.tokens[a.id] = a.text;
            t.ids.putAssumeCapacity(a.text, a.id);
        }
        t.bytes = @splat(null);
        for (0..256) |b| {
            var buf: [6]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{b}) catch unreachable;
            t.bytes[b] = t.ids.get(s);
        }
        // merges: ["a", "b"] pairs or "a b" strings
        t.merges = .empty;
        errdefer t.merges.deinit(gpa);
        var key: std.ArrayList(u8) = .empty;
        defer key.deinit(gpa);
        if (model.get("merges")) |ml| {
            try t.merges.ensureTotalCapacity(gpa, @intCast(ml.array.items.len));
            for (ml.array.items, 0..) |m, rank| {
                var a: []const u8 = undefined;
                var b: []const u8 = undefined;
                switch (m) {
                    .array => |p| {
                        a = p.items[0].string;
                        b = p.items[1].string;
                    },
                    .string => |s| {
                        const sp = std.mem.indexOfScalar(u8, s, ' ') orelse continue;
                        a = s[0..sp];
                        b = s[sp + 1 ..];
                    },
                    else => continue,
                }
                const ia = t.ids.get(a) orelse continue;
                const ib = t.ids.get(b) orelse continue;
                key.clearRetainingCapacity();
                try key.appendSlice(gpa, a);
                try key.appendSlice(gpa, b);
                const merged = t.ids.get(key.items) orelse continue;
                const k = (@as(u64, ia) << 32) | ib;
                if (!t.merges.contains(k)) t.merges.putAssumeCapacity(k, .{ .rank = @intCast(rank), .id = merged });
            }
        }
        t.unk = t.ids.get(if (model.get("unk_token")) |u| u.string else "<unk>") orelse 3;
        t.bos = t.ids.get("<bos>") orelse 2;
        t.eos = t.ids.get("<eos>") orelse 1;
        return t;
    }

    pub fn deinit(t: *Tokenizer) void {
        t.merges.deinit(t.gpa);
        t.ids.deinit(t.gpa);
        t.gpa.free(t.tokens);
        t.gpa.free(t.added);
        t.parsed.deinit();
        t.gpa.destroy(t);
    }

    pub fn text(t: *const Tokenizer, id: u32) []const u8 {
        return if (id < t.tokens.len) t.tokens[id] else "";
    }

    /// BPE over one "▁word" piece.
    fn word(t: *const Tokenizer, gpa: std.mem.Allocator, piece: []const u8, out: *std.ArrayList(u32)) !void {
        var syms: std.ArrayList(u32) = .empty;
        defer syms.deinit(gpa);
        var view = std.unicode.Utf8View.initUnchecked(piece);
        var it = view.iterator();
        var last_unk = false;
        while (it.nextCodepointSlice()) |cp| {
            if (t.ids.get(cp)) |id| {
                try syms.append(gpa, id);
                last_unk = false;
                continue;
            }
            var fallback = true;
            for (cp) |b| if (t.bytes[b] == null) {
                fallback = false;
            };
            if (fallback) {
                for (cp) |b| try syms.append(gpa, t.bytes[b].?);
                last_unk = false;
                continue;
            }
            if (!last_unk) try syms.append(gpa, t.unk);
            last_unk = true;
        }
        // merge the lowest-ranked adjacent pair, leftmost on ties, until none applies
        while (syms.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best: usize = 0;
            var best_id: u32 = 0;
            for (0..syms.items.len - 1) |i| {
                const k = (@as(u64, syms.items[i]) << 32) | syms.items[i + 1];
                if (t.merges.get(k)) |m| if (m.rank < best_rank) {
                    best_rank = m.rank;
                    best = i;
                    best_id = m.id;
                };
            }
            if (best_rank == std.math.maxInt(u32)) break;
            syms.items[best] = best_id;
            _ = syms.orderedRemove(best + 1);
        }
        try out.appendSlice(gpa, syms.items);
    }

    /// Metaspace split of a text segment with no added tokens.
    fn segment(t: *const Tokenizer, gpa: std.mem.Allocator, seg: []const u8, out: *std.ArrayList(u32)) !void {
        if (seg.len == 0) return;
        var s: std.ArrayList(u8) = .empty;
        defer s.deinit(gpa);
        try s.appendSlice(gpa, metaspace);
        var i: usize = 0;
        while (i < seg.len) : (i += 1) {
            if (seg[i] == ' ') try s.appendSlice(gpa, metaspace) else try s.append(gpa, seg[i]);
        }
        // the segment already began with a space: drop the prepended one
        var body: []const u8 = s.items;
        if (std.mem.startsWith(u8, body[metaspace.len..], metaspace)) body = body[metaspace.len..];
        // split before each ▁
        var start: usize = 0;
        var j: usize = metaspace.len;
        while (j < body.len) {
            if (std.mem.startsWith(u8, body[j..], metaspace)) {
                try t.word(gpa, body[start..j], out);
                start = j;
                j += metaspace.len;
            } else j += 1;
        }
        try t.word(gpa, body[start..], out);
    }

    /// Encode text as <bos> pieces <eos>, truncated on the right to max_len ids with the
    /// <eos> kept. Caller owns the result.
    pub fn encode(t: *const Tokenizer, gpa: std.mem.Allocator, s: []const u8, max_len: usize) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        try out.append(gpa, t.bos);
        var seg_start: usize = 0;
        var i: usize = 0;
        while (i < s.len) {
            var hit: ?Added = null;
            for (t.added) |a| if (a.text.len > 0 and std.mem.startsWith(u8, s[i..], a.text)) {
                if (hit == null or a.text.len > hit.?.text.len) hit = a;
            };
            if (hit) |a| {
                try t.segment(gpa, s[seg_start..i], &out);
                try out.append(gpa, a.id);
                i += a.text.len;
                seg_start = i;
            } else i += 1;
        }
        try t.segment(gpa, s[seg_start..], &out);
        try out.append(gpa, t.eos);
        if (max_len >= 2 and out.items.len > max_len) {
            out.shrinkRetainingCapacity(max_len);
            out.items[max_len - 1] = t.eos;
        }
        return out.toOwnedSlice(gpa);
    }
};
