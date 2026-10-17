"""strata-q36 long-context needle test: filler with three facts at 10 / 50 / 85 % depth and a direct question, through
the Qwen chat template (thinking closed); greedy; prints the prompt / decode speed and the needles found.
python q36needle.py TOKENS [ENV=VAL ...]   (run from the Strata tree; Maya's tokenizer reads the GGUF)"""
import os, re, subprocess, sys
sys.path.insert(0, "/media/per/work/llm/maya-q38/tools")
from strata_tokenizer import Tokenizer
G = "/media/per/work/llm/Maya-data/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"
n_tok = int(sys.argv[1])
env = dict(os.environ); env.update(kv.split("=", 1) for kv in sys.argv[2:])
tk = Tokenizer.from_gguf(G)
facts = ["The password for the blue vault is MARIGOLD-4471.", "Captain Ilse Varga was born in the city of Tromsø.",
         "The experiment's final reading was 93.62 kelvin."]
sent = "The committee met to discuss the river, the market and the old library. "
n_s = max(1, n_tok // len(tk.encode(sent)))
chunks = [sent] * n_s
for d, f in zip((0.10, 0.50, 0.85), facts): chunks.insert(int(len(chunks) * d), f + " ")
q = ("".join(chunks) + "\n\nAnswer three questions, one line each: 1) the password for the blue vault, 2) the city where "
     "Captain Ilse Varga was born, 3) the experiment's final reading.")
ids = tk.encode("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
open("/tmp/q36needle.ids", "w").write(",".join(map(str, ids)))
r = subprocess.run(["build/strata-q36", "--native", G, "--max-context", str(len(ids) + 256), "--tokens", "@/tmp/q36needle.ids",
                    "--n", "60"], capture_output=True, text=True, env=env)
m = re.search(r"gen:([0-9 ]+)", r.stdout)
ans = tk.decode([int(x) for x in m.group(1).split()]) if m else ""
ok = [k in ans for k in ("MARIGOLD-4471", "Troms", "93.62")]
sp = re.findall(r"strata-q36: prompt [^\n]*", r.stderr)
print("%d prompt tokens; %s; needles %d / 3 %s" % (len(ids), sp[-1] if sp else r.stderr[-200:], sum(ok), ok))
