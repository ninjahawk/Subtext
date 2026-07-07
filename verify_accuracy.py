# Accuracy audit: does Subtext's live readout path (forward hooks + KV cache +
# transport + unembed) produce the SAME lens readouts as the reference
# implementation (jlens.JacobianLens.apply) from Anthropic's repo?
#
# Run with the Subtext server STOPPED (both need the GPU).

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
# The walkthrough's own example: expects currency-related readouts (lira/euro)
# at mid layers at the position before final.
PROMPT = "Fact: The currency used in the country shaped like a boot is"
LAYERS = [8, 14, 20, 26]
POSITIONS = [-4, -2, -1]
TOPK = 5

# Device: CUDA if present, else Apple Silicon (MPS), else CPU.
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
# bf16 matmuls are slow/patchy on CPU; use fp32 there.
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

print("loading model + lens ...")
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=DTYPE).to(DEVICE)
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(
    LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)

# ---- path A: reference (jlens.apply — no cache, jlens's own forward)
ref_logits, _, input_ids = lens.apply(
    model, PROMPT, layers=LAYERS, positions=POSITIONS)

# ---- path B: Subtext's live path (same hooks + use_cache=True like server.py)
class Catcher:
    def __init__(self, layers, idxs):
        self.acts = {}
        self.h = [layers[i].register_forward_hook(self._mk(i)) for i in idxs]
    def _mk(self, i):
        def hook(_m, _i, out):
            self.acts[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook
    def close(self):
        for x in self.h: x.remove()

catcher = Catcher(model.layers, LAYERS)
with torch.no_grad():
    hf(input_ids=input_ids, use_cache=True)
catcher.close()

seq_len = input_ids.shape[1]
print(f"prompt tokens: {seq_len}\n")
all_ok = True
for layer in LAYERS:
    for pi, pos in enumerate(POSITIONS):
        h = catcher.acts[layer][0, pos].float()
        mine = model.unembed(lens.transport(h, layer)).float().cpu()
        ref = ref_logits[layer][pi]
        top_mine = mine.topk(TOPK).indices.tolist()
        top_ref = ref.topk(TOPK).indices.tolist()
        match = top_mine == top_ref
        cos = torch.nn.functional.cosine_similarity(mine, ref, dim=0).item()
        all_ok &= match
        words = [tok.decode([t]).strip() for t in top_ref]
        print(f"L{layer:>2} pos{pos:>3}  top{TOPK} {'MATCH' if match else 'MISMATCH'}"
              f"  cos={cos:.6f}  ref={words}")
        if not match:
            print(f"      mine={[tok.decode([t]).strip() for t in top_mine]}")

print(f"\n{'PASS: live path == reference' if all_ok else 'FAIL: paths differ'}")
