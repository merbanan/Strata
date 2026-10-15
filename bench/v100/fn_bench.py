"""bench/v100/fn_bench.py (run from the repo root; FN_GGUF = shard 1, FN_PACK = tools/iq_pack.py --gguf FN_GGUF --out <dir>). Flash-Next (Qwen3.8-Flash-Next GSQ-RCO Q2_0) on this card through strata --serve: three short greedy answers (decode
tok/s) and a ~6K-token prompt (prompt tok/s).  FN_PREFILL / FN_RESERVE / FN_CTX / FN_FILL (filler repeats, 400 = ~6K tokens) change the run.  python fnbench.py LABEL [NEW] ; extra engine env from the environment."""
import os, subprocess, sys
sys.path.insert(0, "tools")
import strata_tokenizer as T
P = os.environ.get("FN_PACK", "packs/q2_0")
G = os.environ.get("FN_GGUF", "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf")

tk = T.Tokenizer.from_gguf(G)
label = sys.argv[1]; n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 128
Q = ["Write a Python function that parses a CSV file and returns the average of every numeric column. Explain it.",
     "Explain how a refrigerator works, step by step, for a curious ten year old.",
     "List the first twenty prime numbers and describe the sieve of Eratosthenes."]
FILL = "The committee met to discuss the river, the market and the old library. " * int(os.environ.get("FN_FILL", "400"))
def chat(q): return "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
env = dict(os.environ)
err = "/tmp/fnbench-%s.err" % label
p = subprocess.Popen(["build/strata", "--serve", "--pack", P, "--native", G, "--ple-gguf", G.replace("00001-of", "00002-of"), "--expert-profile",
                      "data/expert-profile.bin", "--expert-cache", "auto", "--max-context", os.environ.get("FN_CTX", "8192"), "--kv", "int8", "--spec", "2", "--prefill", os.environ.get("FN_PREFILL", "auto"),
                      "--vram-reserve-mib", os.environ.get("FN_RESERVE", "2048")],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(err, "w"), text=True, env=env)
for line in p.stdout:
    if line.startswith(("READY", "ERR")):
        if line.startswith("ERR"): print(line.strip())
        break
def gen(ids, n):
    p.stdin.write("GEN %d temperature=0 ckpt=0 %s\n" % (n, ",".join(map(str, ids)))); p.stdin.flush()
    toks = []
    for line in p.stdout:
        if line.startswith("T "): toks.append(int(line.split()[1]))
        elif line.startswith(("DONE", "ERR")):
            if line.startswith("ERR"): print(line.strip()); return toks, 0, 0.0, 1.0
            f = line.split(); return toks, int(f[2]), float(f[3]), float(f[4])
gen(tk.encode(chat("Hello"), parse_special=True), 8)   # warm-up
out = []
for q in Q:
    toks, n, pms, dms = gen(tk.encode(chat(q), parse_special=True), n_new)
    out.append(toks)
    print("%s decode: %d tokens, %.1f tok/s" % (label, len(toks), (len(toks) - 1) * 1000 / max(dms, 1e-9)), flush=True)
ids = tk.encode(chat(FILL + "\nSummarize the text above in one sentence."), parse_special=True)
toks, n, pms, dms = gen(ids, 8)
print("%s prompt: %d tokens in %.0f ms = %.1f tok/s" % (label, n, pms, n * 1000 / max(pms, 1e-9)))
import hashlib; print("%s tokens md5 %s" % (label, hashlib.md5(repr(out).encode()).hexdigest()[:12]))
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=120)
