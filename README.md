# Subtext

**Watch a language model think, live.** Subtext is a local visualization of the
J-space — the emergent "global workspace" Anthropic described in
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(July 2026). You chat with a small local model and watch its silent thoughts
form in real time: while it reads your message, while it reasons, and in the
moments before words leave its mouth.

The readout is the [Jacobian lens](https://github.com/anthropics/jacobian-lens):
for a residual-stream activation at any layer, it answers *"which vocabulary
words is this internal state disposed to make the model say, now or later?"*
Subtext applies the lens at nine depths on every token and renders the result
as an instrument you can read.

## What you're looking at

- **Every glowing word is a silent thought** — an internal activation pushing
  the model toward saying that word. It is not text the model wrote.
- **Vertical position = layer.** Thoughts near the top live in early layers
  (perception); they sink toward the bottom rail as they approach speech.
  Multi-step reasoning happens in between — ask a two-hop question
  ("the currency of the country shaped like a boot?") and the intermediate
  step (*Italy*) lights up mid-network before the answer (*euro*) forms below it.
- **Size and brightness are absolute strength.** A faint thought really is
  faint. Amber = while reading you; blue = while composing its reply.
- **Hover any word** for its layer-by-layer trace; **click** for an inspector
  with strength history.
- The right panel is the paper trail: conversation, counters, a live
  top-of-mind ranking, and a per-token ledger of everything the lens saw.

Try: `Is this correct? 12 + 5 = 1` — and watch *incorrect* ignite in the
middle layers while the model is still reading the equation, several tokens
before it begins to answer.

## Requirements

- NVIDIA GPU with ~10 GB VRAM (developed on an RTX 5070, 12 GB)
- Python 3.11+, CUDA build of PyTorch
- ~9 GB disk for Qwen3.5-4B + the pre-fitted lens (downloaded on first run)

## Install & run

```bash
git clone https://github.com/ninjahawk/Subtext
cd Subtext
pip install -r requirements.txt   # torch: install your CUDA build first if needed
python server.py                  # first run downloads model + lens, builds a token mask
# → http://localhost:8765
```

Windows note: run `python -u -X utf8 server.py` (or just `start.bat`).

## How it works

```
browser (index.html, one file, no build)  ⇐ websocket ⇐  server.py
    Qwen3.5-4B (bf16, HF transformers, KV cache)
    + pre-fitted Jacobian lens (neuronpedia/jacobian-lens, rev qwen-n1000)
    per token: residual hooks at 9 layers → J_l transport → unembed
             → softmax → top-k word-start tokens → frame {word, p, depth, profile}
```

Two phases per exchange: a **reading** pass over your message (one prefill,
lens at every position), then **thinking** frames streamed during generation.
The lens costs a few small matmuls per token, so it runs at full generation
speed.

### Display filtering

Raw lens top-k is full of punctuation and BPE fragments ("itude" is a piece of
*certitude*, not a thought). Subtext restricts display to word-start tokens
(the paper's own `mask_display` filter, tightened), and maps strength through
an absolute scale — the display never inflates a weak readout. Ranks and
probabilities are computed over the full vocabulary first, so filtering never
changes what the lens actually said, only what is legible.

## Accuracy

`verify_accuracy.py` compares this live path (forward hooks + KV cache)
against the reference `JacobianLens.apply()` from Anthropic's repo, token by
token. On the walkthrough prompt, all layer/position top-5 readouts match
exactly (cosine ≥ 0.99998), and reproduce the paper's two-hop signature
(*Italy* at layer 20, *euros* at layer 26, before any output). Run it yourself
with the server stopped.

## Honest limitations

Inherited from the method (see §"Limitations" of the paper): the lens only
reads concepts that are single tokens in the model's vocabulary; it
approximately captures the workspace, not all of it; and none of this
demonstrates subjective experience — the paper is explicit that the workspace
shows *access*-consciousness-like function, not feeling.

## Credits

- Method & reference implementation: [Anthropic — jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache 2.0)
- Pre-fitted lens weights: [Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens)
- Model: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

Apache 2.0. Built by [@ninjahawk](https://github.com/ninjahawk) with Claude.
