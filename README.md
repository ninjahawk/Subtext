<div align="center">

# Subtext

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-Qwen3.5--4B-6cb8e0)](https://huggingface.co/Qwen/Qwen3.5-4B)
[![Lens](https://img.shields.io/badge/Lens-qwen--n1000-e8a13c)](https://huggingface.co/neuronpedia/jacobian-lens)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

*A real-time instrument for observing the verbal workspace of a language model<br>as it reads, reasons, and speaks.*

![Subtext demo](media/demo.gif)

**[▶ Full demo video](media/demo.mp4)** · **[📄 The paper this builds on](https://transformer-circuits.pub/2026/workspace/index.html)** · **[🔬 Reference implementation](https://github.com/anthropics/jacobian-lens)**

</div>

---

## Overview

Recent work from Anthropic identified a small set of internal representations in
language models — the *J-space* — that behaves like a global workspace: its
contents can be verbally reported by the model, deliberately modulated, and are
causally used for multi-step reasoning, while the surrounding majority of neural
activity remains inaccessible to report. The identification tool is the
Jacobian lens, which transports a residual-stream activation at any layer into
the final-layer basis and decodes it through the model's own unembedding,
answering: *which vocabulary words is this internal state disposed to produce,
now or later?*

Subtext applies that method continuously during live conversation with a local
model. On every token — both while the model ingests the user's message and
while it generates its reply — the lens is read at nine depths and the result
is rendered as it happens. The intermediate steps of the model's computation
become directly watchable: verdicts form during reading, several tokens before
any output; planned words hold at high strength while unrelated tokens are
being emitted; two-hop questions surface their unspoken middle term.

A representative example: given `Is this correct? 12 + 5 = 1`, the workspace
readout shows *incorrect* active in mid layers while the equation is still
being read, and given the country-shaped-like-a-boot currency question, *Italy*
appears at layer 20 and *euros* at layer 26 before generation begins —
reproducing the two-hop signature reported in the paper.

## Reading the display

- **Each rendered word is a lens readout, not model output.** It indicates an
  internal activation disposing the model toward that word.
- **Vertical position corresponds to layer.** Early layers (perception) are at
  the top; readouts approach the bottom rail as they approach emission.
- **Size and opacity encode absolute readout strength.** The display applies a
  fixed monotone mapping from lens probability; weak readouts are rendered
  weak. Amber marks readouts taken while reading the user; blue while
  generating.
- **Hover** shows a word's per-layer activation profile; **click** opens an
  inspector with peak strength, mean depth, and strength history.
- The right panel records everything the canvas curates: the conversation, a
  live ranking of currently-active readouts, and a per-token ledger.

## Method

```
browser (single HTML file)  ⇐ websocket ⇐  server.py
    Qwen3.5-4B (bf16, HF transformers, KV cache)
    pre-fitted Jacobian lens: neuronpedia/jacobian-lens, revision qwen-n1000
    per token: residual hooks at 9 layers → J_l transport → unembed
             → full-vocabulary softmax → word-start top-k → frame
```

Each exchange has two phases. A single prefill pass covers the user's message,
with lens readouts taken at every position (the *reading* phase); generation
then proceeds token-by-token with a KV cache, reading the lens at the newest
position each step (the *thinking* phase). The lens adds a per-layer
matrix-vector product and an unembedding per token, so streaming runs at the
model's native generation speed.

**Display filtering.** Raw lens top-k contains punctuation and BPE
continuation fragments (e.g. *itude*, from *cert‑itude*), which are not
meaningful as readouts. Display is restricted to word-initial vocabulary
tokens, following the reference implementation's `mask_display` with a
stricter word-start criterion. Probabilities are computed over the full
vocabulary before any filtering, so filtering affects legibility only, never
the readout itself.

## Validation

`verify_accuracy.py` compares this implementation's live path (forward hooks,
KV cache enabled) against the reference `JacobianLens.apply()` on identical
inputs. Across 4 layers × 3 positions on the walkthrough prompt, top-5
readouts match exactly, with cosine similarity ≥ 0.99998 between logit
vectors, and reproduce the expected two-hop intermediates. The audit can be
re-run at any time with the server stopped.

## Setup

Requirements: an NVIDIA GPU with ~10 GB of VRAM, Python 3.11+, and a CUDA
build of PyTorch. First launch downloads the model and lens (~9 GB total) and
builds a display-token mask (~1 minute, cached).

```bash
git clone https://github.com/ninjahawk/Subtext
cd Subtext
pip install -r requirements.txt
python server.py
# → http://localhost:8765
```

On Windows, run `python -u -X utf8 server.py`, or use `start.bat`.

## Limitations

The instrument inherits the method's limitations. The lens reads only concepts
that correspond to single vocabulary tokens; multi-token concepts are invisible
or fragmentary. It approximately captures the workspace identified in the
paper, not the entirety of the model's internal state, and layers below the
fitted range are not observed. Interpretation should also respect the paper's
own framing: workspace readouts demonstrate functional availability of
information for report and reasoning; they do not demonstrate subjective
experience.

## Acknowledgements

The method and reference implementation are by Anthropic
([jacobian-lens](https://github.com/anthropics/jacobian-lens), Apache 2.0).
Pre-fitted lens weights are published by
[Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens). The model is
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Subtext is an
independent project and is not affiliated with Anthropic.

Licensed under Apache 2.0.
