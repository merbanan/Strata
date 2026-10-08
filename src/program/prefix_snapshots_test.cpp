// src/program/prefix_snapshots_test.cpp - the O06 selectors and the protected retention rule (header-only, CPU).
#include "strata/program/prefix_snapshots.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace ps = strata::program::prefix_snapshots;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
constexpr int64_t S = 248045, E = 248046, NL = 198;

// <S>system ... <E>\n <S>user <body> <E>\n <S>assistant\n  with the given lengths; returns the last <S> in turn_at
std::vector<int64_t> chat(int64_t sys, std::vector<int64_t> bodies, int64_t& turn_at) {
    std::vector<int64_t> ids;
    auto msg = [&](int64_t len) {
        ids.push_back(S);
        for (int64_t i = 0; i < len; ++i) ids.push_back(1000 + (int64_t) (ids.size() % 7));
        ids.push_back(E);
        ids.push_back(NL);
    };
    msg(sys);
    for (int64_t b : bodies) msg(b);
    turn_at = (int64_t) ids.size();
    ids.push_back(S);
    ids.push_back(77);
    ids.push_back(NL);
    return ids;
}
}  // namespace

int main() {
    std::printf("prefix_snapshots_test\n");
    int64_t t = 0;
    {
        const auto ids = chat(3000, {2000}, t);   // one long user message after the system prompt
        const int64_t open = 3000 + 3;            // the user turn's <S>
        check(ps::message_start(ids, 0, t, S, 512) == open, "message start: the last message's <|im_start|>");
        check(ps::message_start(ids, 2900, t, S, 512) == -1, "message start: too few fresh tokens before it");
        check(ps::content_end(ids, 0, t, S, E, 4, 512) == t - 2 - 4, "content end: margin before the closing <|im_end|>");
        check(ps::content_end(ids, 0, t, S, E, 4, 2100) == -1, "content end: a message shorter than min_fresh");
        check(ps::content_end(ids, open + 1500, t, S, E, 4, 512) == -1, "content end: resumed inside the message");
        check(ps::content_end(ids, 0, t, S, -1, 4, 512) == -1, "content end: no end token configured");
    }
    {
        int64_t t2 = 0;
        auto ids = chat(100, {50}, t2);
        ids[(size_t) t2 - 2] = 5;   // no <|im_end|> right before the header
        check(ps::content_end(ids, 0, t2, S, E, 0, 1) == -1, "content end: no closing token near the header");
        check(ps::message_start(ids, 0, 0, S, 1) == -1 && ps::message_start(ids, 0, 5, S, 1) == -1,
              "message start: an invalid turn position");
    }
    {
        // the role check: chat() puts token 1000 + k after each <S>; make the last message a user / tool / assistant turn
        constexpr int64_t U = 846, TR = 248066, A = 74455;
        int64_t tu = 0;
        auto ids = chat(3000, {2500}, tu);
        const size_t open = 3000 + 3;
        ids[open + 1] = U;
        check(ps::long_user_message(ids, tu, S, U, TR, 2048), "long user message: a 2.5K user turn qualifies");
        check(!ps::long_user_message(ids, tu, S, U, TR, 3000), "long user message: shorter than msg_min");
        ids[open + 2] = NL;
        ids[open + 3] = TR;
        check(!ps::long_user_message(ids, tu, S, U, TR, 2048), "long user message: a tool result (<tool_response>)");
        check(ps::long_user_message(ids, tu, S, U, -1, 2048), "long user message: tool check off");
        ids[open + 3] = 5;
        ids[open + 1] = A;
        check(!ps::long_user_message(ids, tu, S, U, TR, 2048), "long user message: an assistant turn");
        ids[open + 1] = U;
        check(!ps::long_user_message(ids, 0, S, U, TR, 1) && !ps::long_user_message(ids, 5, S, U, TR, 1),
              "long user message: an invalid turn position");
        // the message's own length counts, also when most of it was read before (resumed inside it)
        check(ps::long_user_message(ids, tu, S, U, TR, 2400), "long user message: own length, not fresh tokens");
    }
    {
        // root, two older leaves, then this request's message + content checkpoints (stamps 5, 6) and its turn (7)
        const std::vector<uint64_t> st = {1, 2, 3, 5, 6, 7};
        check(ps::eviction_victim(st.data(), st.size(), 5, nullptr, 4) == 1, "protected: an older leaf goes first");
        const std::vector<uint64_t> mine = {1, 5, 6, 7};
        check(ps::eviction_victim(mine.data(), mine.size(), 3, nullptr, 4) == 1,
              "protected: only the request's own left - the plain LRU rule");
        const bool tail[6] = {false, false, true, false, false, false};
        check(ps::eviction_victim(st.data(), st.size(), 5, tail, 4) == 2, "protected: an older tail item before LRU");
        check(ps::eviction_victim(st.data(), st.size(), 1, nullptr, 4) == 0, "one-slot budget: the oldest (no pin)");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
