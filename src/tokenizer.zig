//! Text to token ids, the way whisper.cpp does it for prompts: split into GPT-2 style
//! words (a leading space stays with its word, punctuation runs are their own words,
//! common contractions split off), then cover each word greedily with the longest
//! vocabulary entry that matches. Not true byte-pair encoding, but it agrees with it
//! on ordinary words and names, which is all a prompt or a hotword needs.
const std = @import("std");
const model = @import("model.zig");

fn isAlpha(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c >= 0x80; // treat UTF-8 bytes as letters
}

/// Append the words of `text` to `out` as slices of `text`.
fn split(gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    const contractions = [_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
    while (i < text.len) {
        var matched = false;
        for (contractions) |c| {
            if (std.mem.startsWith(u8, text[i..], c)) {
                try out.append(gpa, text[i..][0..c.len]);
                i += c.len;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        const start = i;
        var j = i;
        if (text[j] == ' ') j += 1;
        if (j < text.len and isAlpha(text[j])) {
            while (j < text.len and isAlpha(text[j])) j += 1;
        } else if (j < text.len and std.ascii.isDigit(text[j])) {
            while (j < text.len and std.ascii.isDigit(text[j])) j += 1;
        } else if (j < text.len and !std.ascii.isWhitespace(text[j])) {
            while (j < text.len and !std.ascii.isWhitespace(text[j]) and !isAlpha(text[j]) and !std.ascii.isDigit(text[j])) j += 1;
        } else {
            // whitespace run (the optional space matched nothing after it)
            j = start;
            while (j < text.len and std.ascii.isWhitespace(text[j])) j += 1;
        }
        if (j == start) j = start + 1;
        try out.append(gpa, text[start..j]);
        i = j;
    }
}

/// Tokenize `text`; the result is owned by the caller. Unknown bytes are skipped.
pub fn encode(gpa: std.mem.Allocator, m: *const model.Model, text: []const u8) ![]u32 {
    var words: std.ArrayList([]const u8) = .empty;
    defer words.deinit(gpa);
    try split(gpa, text, &words);
    var tokens: std.ArrayList(u32) = .empty;
    errdefer tokens.deinit(gpa);
    for (words.items) |word| {
        if (word.len == 0) continue;
        var i: usize = 0;
        while (i < word.len) {
            var j = word.len;
            var found = false;
            while (j > i) : (j -= 1) {
                if (m.tokenId(word[i..j])) |id| {
                    try tokens.append(gpa, id);
                    i = j;
                    found = true;
                    break;
                }
            }
            if (!found) i += 1;
        }
    }
    return tokens.toOwnedSlice(gpa);
}

test "split keeps the space with the word and separates punctuation" {
    const gpa = std.testing.allocator;
    var words: std.ArrayList([]const u8) = .empty;
    defer words.deinit(gpa);
    try split(gpa, "Carl's dog, \"Donut\" ran.", &words);
    const want = [_][]const u8{ "Carl", "'s", " dog", ",", " \"", "Donut", "\"", " ran", "." };
    try std.testing.expectEqual(want.len, words.items.len);
    for (want, words.items) |w, g| try std.testing.expectEqualStrings(w, g);
}
