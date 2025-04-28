"""
Monkey-patch dstc12.utils.get_llm so every baseline script
uses local Gemma-3-27B (llama-cpp) instead of HF Mistral.
"""

import os, sys, re, importlib, importlib.util
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama

# ── locate / download .gguf ─────────────────────────────
GGUF = os.getenv("GEMMA_PATH")
if GGUF is None:
    repo_id = "google/gemma-3-27b-it-qat-q4_0-gguf"
    fn      = "gemma-3-27b-it-q4_0.gguf"
    if (tok := os.getenv("HF_TOKEN")):
        login(token=tok, add_to_git_credential=True)
    GGUF = hf_hub_download(repo_id=repo_id, filename=fn, resume_download=True)

# ── load model ──────────────────────────────────────────
gemma = Llama(model_path=GGUF, n_ctx=20480, n_gpu_layers=-1, verbose=False)
SYS = (
  "You are an expert conversation analyst.\n"
  "Reply in exactly TWO XML lines:\n"
  "<theme_label>Short Title Here</theme_label>\n"
  "<theme_label_explanation>One short sentence explaining why.</theme_label_explanation>\n"
  "• Theme label ≤5 words, Title Case.\n"
  "• No other text."
)

def _generate(text: str) -> str:
    out = gemma.create_chat_completion(
        messages=[{"role": "system", "content": SYS},
                  {"role": "user",   "content": text.strip()}],
        temperature=0.3, max_tokens=80)
    return out["choices"][0]["message"]["content"].strip()

# LangChain-compatible wrapper
class GemmaRunnable:
    def __call__(self, prompt):
        return _generate(str(prompt))        # works for str / PromptValue / dict

# ── patch get_llm ───────────────────────────────────────
spec = importlib.util.find_spec("dstc12.utils")
if spec is None:
    sys.exit("Run  `source set_paths.sh`  so PYTHONPATH includes ./src")
dst_utils = importlib.import_module("dstc12.utils")
dst_utils.get_llm = lambda *_args, **_kw: GemmaRunnable()

print("✅  Patched dstc12.utils.get_llm → local Gemma-3-27B")

