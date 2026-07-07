# Subtext — live Jacobian-lens thought streaming for a local model.
#
# Loads Qwen3.5-4B + Anthropic/Neuronpedia's pre-fitted Jacobian lens, serves
# a chat websocket. For every token — both while READING your message and while
# GENERATING its reply — it reads the residual stream at a spread of layers,
# transports each through J_l into the final-layer basis, decodes to vocabulary
# words, and streams the top "silent words" to the browser.
#
# Run:  python server.py   → http://localhost:8765

import asyncio
import json
import pathlib
import re

import torch
import transformers
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

import jlens
from jlens.vis import _meaningful_token_mask

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"

HERE = pathlib.Path(__file__).parent
PORT = 8765
TOPK_PER_LAYER = 8
MAX_WORDS_PER_FRAME = 14
# Fractions of network depth to read out at. Sparse early (reading), dense in
# the middle (the workspace), one near the end (about to speak).
LAYER_FRACS = [0.12, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.93]

# Device: CUDA if present, else Apple Silicon (MPS), else CPU.
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
# bf16 matmuls are slow/patchy on CPU; use fp32 there.
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

print(f"[subtext] loading {MODEL_NAME} ({'bf16' if DTYPE == torch.bfloat16 else 'fp32'}, {DEVICE}) ...")
hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=DTYPE
).to(DEVICE)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf_model, tokenizer)

print("[subtext] loading pre-fitted Jacobian lens ...")
lens = jlens.JacobianLens.from_pretrained(
    LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
)

# Pick actual fitted layers closest to the requested depth fractions.
n_layers = model.n_layers
_wanted = [round(f * (n_layers - 1)) for f in LAYER_FRACS]
VIZ_LAYERS = sorted({min(lens.source_layers, key=lambda s, w=w: abs(s - w)) for w in _wanted})
LAYER_DEPTH = {l: l / (n_layers - 1) for l in VIZ_LAYERS}
print(f"[subtext] reading layers {VIZ_LAYERS} of {n_layers}")

# Keep only the layers we use, resident on the GPU (transport is then free).
lens.jacobians = {l: lens.jacobians[l].to(DEVICE) for l in VIZ_LAYERS}

# Display mask: word-like tokens only (the paper's own filter), further
# restricted to ASCII so the stream reads in English.
_mask_path = HERE / "token_mask.pt"
vocab_size = hf_model.get_output_embeddings().weight.shape[0]
if _mask_path.exists():
    display_mask = torch.load(_mask_path, weights_only=True).to(DEVICE)
else:
    print("[subtext] building word-like token mask (one-time, ~1 min) ...")
    display_mask = _meaningful_token_mask(tokenizer, vocab_size, torch.device("cpu"))
    for tid in display_mask.nonzero().flatten().tolist():
        raw = tokenizer.decode([tid])
        s = raw.strip()
        # Word-START tokens only (leading space in Qwen's vocab): subword
        # continuations like "itude" (from cert-itude) are fragments, not
        # thoughts, and displaying them is misleading.
        if not (raw.startswith(" ") and s.isascii()
                and re.match(r"^[A-Za-z][A-Za-z'\-]+$", s) and len(s) > 2):
            display_mask[tid] = False
    torch.save(display_mask, _mask_path)
    display_mask = display_mask.to(DEVICE)
print(f"[subtext] display vocabulary: {int(display_mask.sum())} word tokens")


class ResidualCatcher:
    """Forward hooks on the chosen decoder blocks; keeps each block's output
    hidden states from the most recent forward."""

    def __init__(self, layers, indices):
        self.acts: dict[int, torch.Tensor] = {}
        self._handles = [
            layers[i].register_forward_hook(self._make(i)) for i in indices
        ]

    def _make(self, i):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.acts[i] = h.detach()
        return hook

    def close(self):
        for h in self._handles:
            h.remove()


@torch.no_grad()
def readout(residuals: dict[int, torch.Tensor], position: int) -> list[dict]:
    """Aggregate lens top-k across layers into one frame's thought list.

    residuals: {layer: [1, seq, d_model]}; position indexes into seq.
    Returns [{w, p, d}]: word, strength (max prob across layers, normalised
    per frame client-side), depth (probability-weighted mean layer fraction).
    """
    words: dict[str, dict] = {}
    for layer in VIZ_LAYERS:
        h = residuals[layer][0, position].float()
        transported = lens.transport(h, layer)
        logits = model.unembed(transported).float()
        probs = torch.softmax(logits, -1)
        probs = probs.masked_fill(~display_mask, 0.0)
        top = probs.topk(TOPK_PER_LAYER)
        for p, tid in zip(top.values.tolist(), top.indices.tolist()):
            if p <= 0.0:
                continue
            w = tokenizer.decode([tid]).strip().lower()
            e = words.setdefault(w, {"p": 0.0, "pd": 0.0, "psum": 0.0, "prof": {}})
            e["p"] = max(e["p"], p)
            e["pd"] += p * LAYER_DEPTH[layer]
            e["psum"] += p
            e["prof"][layer] = max(e["prof"].get(layer, 0.0), p)
    out = [
        {
            "w": w, "p": round(e["p"], 5), "d": round(e["pd"] / e["psum"], 3),
            "prof": [[l, round(p, 5)] for l, p in sorted(e["prof"].items())],
        }
        for w, e in words.items()
    ]
    out.sort(key=lambda x: -x["p"])
    return out[:MAX_WORDS_PER_FRAME]


def _template_ids(messages: list[dict], add_gen: bool) -> torch.Tensor:
    out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_gen, return_tensors="pt",
        enable_thinking=False,
    )
    return out if isinstance(out, torch.Tensor) else out["input_ids"]


def encode_chat(messages: list[dict]) -> tuple[torch.Tensor, int]:
    """Tokenize the conversation; return (input_ids, index where the newest
    user message starts) so the reading phase covers just what was typed."""
    ids = _template_ids(messages, True).to(model.input_device)
    prev = (_template_ids(messages[:-1], False)
            if len(messages) > 1 else torch.zeros(1, 0, dtype=torch.long))
    return ids, min(prev.shape[1], ids.shape[1] - 1)


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps({
        "type": "hello", "model": MODEL_NAME.split("/")[-1],
        "n_layers": n_layers, "layers": VIZ_LAYERS,
    }))
    try:
        while True:
            req = json.loads(await ws.receive_text())
            if req.get("type") != "chat":
                continue
            messages = req["messages"]
            max_new = min(int(req.get("max_tokens", 220)), 512)
            temperature = float(req.get("temperature", 0.8))

            input_ids, read_from = encode_chat(messages)
            catcher = ResidualCatcher(model.layers, VIZ_LAYERS)
            try:
                # ---- phase 1: READING — one prefill pass, lens at each
                # position of the user's newest message. (use_cache without an
                # explicit cache object: hybrid-attention models build their own)
                with torch.no_grad():
                    out = hf_model(input_ids=input_ids, use_cache=True)
                cache = out.past_key_values
                resids = dict(catcher.acts)
                for pos in range(read_from, input_ids.shape[1]):
                    tok_text = tokenizer.decode([input_ids[0, pos].item()])
                    frame = {
                        "type": "frame", "phase": "reading", "tok": tok_text,
                        "thoughts": readout(resids, pos),
                    }
                    await ws.send_text(json.dumps(frame))
                    await asyncio.sleep(0)  # yield to the event loop

                # ---- phase 2: THINKING/SPEAKING — token-by-token decode with
                # KV cache; lens on the newest position each step.
                eos_ids = hf_model.generation_config.eos_token_id
                eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
                logits = out.logits[:, -1]
                reply_ids: list[int] = []
                for _ in range(max_new):
                    probs = torch.softmax(logits.float() / temperature, -1)
                    # top-p 0.95
                    sp, si = probs.sort(descending=True)
                    keep = sp.cumsum(-1) - sp < 0.95
                    sp = sp * keep
                    nxt = si[0, torch.multinomial(sp[0] / sp.sum(), 1)].item()
                    if nxt in eos_ids:
                        break
                    reply_ids.append(nxt)
                    with torch.no_grad():
                        out = hf_model(
                            input_ids=torch.tensor([[nxt]], device=model.input_device),
                            past_key_values=cache, use_cache=True,
                        )
                    cache = out.past_key_values
                    logits = out.logits[:, -1]
                    frame = {
                        "type": "frame", "phase": "thinking",
                        "out": tokenizer.decode([nxt]),
                        "thoughts": readout(catcher.acts, -1),
                    }
                    await ws.send_text(json.dumps(frame))
                    await asyncio.sleep(0)

                await ws.send_text(json.dumps({
                    "type": "done",
                    "text": tokenizer.decode(reply_ids, skip_special_tokens=True),
                }))
            finally:
                catcher.close()
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    print(f"[subtext] ready -> http://localhost:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
