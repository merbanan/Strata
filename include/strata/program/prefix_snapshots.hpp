// include/strata/program/prefix_snapshots.hpp - O06, opt-in (STRATA_PREFIX_SNAPSHOTS=1): extra conversation
// checkpoints at stable points inside the new part of a prompt, so a request that diverges there resumes close to
// the divergence instead of at the previous turn boundary.
//
// The serve loop already checkpoints the end of the system prompt (--prompt-cache-root), the last turn boundary
// (the new assistant header) and every --prompt-cache-every fresh tokens.  Two cases it misses:
//   * the last message is EDITED: the deepest checkpoint is the previous turn's boundary, so the previous assistant
//     answer and the message are read again -> `message`: the <|im_start|> that opens the last message;
//   * the last message is EXTENDED (the same user turn sent again with more text): the turn checkpoint holds the
//     message's closing <|im_end|>, so nothing of the message matches -> `content`: a few tokens (`margin`) before
//     the closing <|im_end|>, so a re-tokenized join still matches.
// Both are only taken when the last message is a long user message (`long_user_message`: not a tool result or an
// assistant turn, which agent loops append and never revisit) and they keep at least `min_fresh` tokens a later
// request would otherwise read again; any other request gets no extra checkpoint and no extra prompt part.
//
// The drafter (MTP) cell at L-1 was computed with token L of the prompt that saved the checkpoint (MTP drafts
// from h_{L-1} and the NEXT token), so a checkpoint at an arbitrary position is only mounted when the new prompt
// has that same token at L (`next`); at a turn boundary that token is always <|im_start|>.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "strata/program/conv_cache.hpp"

namespace strata::program::prefix_snapshots {

/// The <|im_start|> that opens the last message before the new assistant header at `turn_at`, when at least
/// `min_fresh` freshly read tokens (from `from`) lie before it; else -1.
inline int64_t message_start(const std::vector<int64_t>& ids, int64_t from, int64_t turn_at, int64_t turn_token,
                             int64_t min_fresh) {
    if (from < 0 || turn_token < 0 || min_fresh < 1 || turn_at <= from || turn_at >= (int64_t) ids.size() ||
        ids[(size_t) turn_at] != turn_token) return -1;
    for (int64_t i = turn_at - 1; i > from; --i)
        if (ids[(size_t) i] == turn_token) return i - from >= min_fresh ? i : -1;
    return -1;
}

/// `margin` tokens before the <|im_end|> that closes the last message (it must sit within the 4 tokens before
/// `turn_at`), when that message has at least `min_fresh` tokens past max(from, its opening <|im_start|>); else -1.
inline int64_t content_end(const std::vector<int64_t>& ids, int64_t from, int64_t turn_at, int64_t turn_token,
                           int64_t end_token, int64_t margin, int64_t min_fresh) {
    if (from < 0 || turn_token < 0 || end_token < 0 || margin < 0 || min_fresh < 1 || turn_at <= from ||
        turn_at >= (int64_t) ids.size() || ids[(size_t) turn_at] != turn_token) return -1;
    int64_t close = -1;
    for (int64_t i = turn_at - 1; i > from && i >= turn_at - 4; --i)
        if (ids[(size_t) i] == end_token) { close = i; break; }
    if (close < 0) return -1;
    int64_t open = from;
    for (int64_t i = close - 1; i > from; --i)
        if (ids[(size_t) i] == turn_token) { open = i; break; }
    const int64_t L = close - margin;
    return L - open >= min_fresh ? L : -1;
}

/// Whether the last message before the new assistant header at `turn_at` is a long USER message - the only kind a
/// client edits or sends again extended; agent loops append tool results and assistant turns and never revisit them,
/// so checkpoints inside those cost a sync + host copy and a prompt split for nothing.  The message opens with
/// <|im_start|> `user_token` (searched back over the whole prompt, the cached part included), is not a tool result
/// (the template renders those as a user turn whose content starts with `tool_response_token`, within the 3 tokens
/// after the role), and spans at least `msg_min` tokens from its <|im_start|> to `turn_at`.
inline bool long_user_message(const std::vector<int64_t>& ids, int64_t turn_at, int64_t turn_token, int64_t user_token,
                              int64_t tool_response_token, int64_t msg_min) {
    if (turn_token < 0 || user_token < 0 || turn_at < 1 || turn_at >= (int64_t) ids.size() ||
        ids[(size_t) turn_at] != turn_token) return false;
    int64_t open = -1;
    for (int64_t i = turn_at - 1; i >= 0; --i)
        if (ids[(size_t) i] == turn_token) { open = i; break; }
    if (open < 0 || turn_at - open < std::max<int64_t>(msg_min, 1) || ids[(size_t) open + 1] != user_token) return false;
    if (tool_response_token >= 0)
        for (int64_t i = open + 2; i < turn_at && i <= open + 4; ++i)
            if (ids[(size_t) i] == tool_response_token) return false;
    return true;
}

/// The retention rule with the request's own checkpoints protected: a checkpoint stamped after `protect_after`
/// (created or mounted by the running request) only leaves when nothing older can.  Among the older ones the
/// --prompt-cache-tail item goes first (`tail`, may be null), then the least recently used; never the root.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, const bool* tail, uint64_t protect_after) {
    if (cap >= 2 && n >= 2) {
        size_t v = 0;
        for (size_t i = 1; i < n; ++i) {
            if (stamps[i] > protect_after) continue;
            if (tail != nullptr && tail[i]) return i;
            if (v == 0 || stamps[i] < stamps[v]) v = i;
        }
        if (v != 0) return v;
    }
    return conv_cache::eviction_victim(stamps, n, cap, tail);
}

}  // namespace strata::program::prefix_snapshots
