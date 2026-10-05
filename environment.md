# Environment

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

The analysis scripts -- the ones that rebuild the paper's tables and figures from the
stored outputs in `artifacts/` -- run on a laptop in a few minutes. `torch` is in the
requirements because `nashlib` imports it at module level, but those scripts never move
a tensor to a device; a CPU-only build is enough for them.

## What the runs need

Running a model needs a CUDA GPU. Memory is set by the largest model, not by the
method: the 7-9B autoregressive baselines need about 30 GB in float32, the masked
encoders under 4 GB. The experiments were run on NVIDIA A100, A40, A6000, B200 and RTX
PRO 6000 Blackwell cards.

**mmBERT-base on CLAPNQ: pass `--batch-size 16`.** mmBERT-base has a 256k-entry
vocabulary, and the masked decoder materializes logits at every position of every
sequence in a scan, so its longest CLAPNQ canvases take a smaller batch than the
default of 64.

**Host memory.** `from_pretrained` materializes a model's float32 weights in host RAM
before moving them to the GPU, so a 7B baseline uses about 30 GB of RAM on the node.
Setting

```bash
export NASHLIB_LOAD_ON_DEVICE=1      # needs `pip install accelerate`
```

makes the autoregressive decoder load each tensor straight onto the device instead.
The weights are the same either way, so the outputs are too.

Versions the code has been run under:

| | version |
|---|---|
| Python | 3.11 and 3.13 |
| PyTorch | 2.9-2.12, CUDA 12.x and 13.0 |
| transformers | 4.48-5.12 |
| rouge-score | 0.1.2 |

## Numerics

All models run in float32 with TF32 disabled and decoding is greedy, so all decoding
procedures are deterministic under the reported experimental configuration.

TF32 rounds float32 matmul mantissas to 10 bits on Ampere and later. It is invisible
in a config dump and it changes which token wins an argmax when two logits are close
-- exactly the situation a Nash gap of 1e-3 lives in. So every run calls
`nashlib.numerics.configure(device)`, which disables TF32 on matmul and cuDNN and then
asserts it: a 2048x2048 float32 product must agree with a float64 reference to a
relative error below 1e-5. With TF32 active the error is about 1e-3.

Attention uses the SDPA implementation everywhere but one. The exception is BLOOM-3B:
`transformers` ships no SDPA kernel for that architecture, so it runs under eager
attention. That is recorded on its `ModelSpec` as `attention="eager"`, so every script
picks it up from the same place.

## Model downloads

All 40 checkpoints in `nashlib/registry.py` are pinned to a commit hash, so
`from_pretrained` fetches the same weights regardless of what the branch has moved to
since.

Set `HF_HOME` to a directory with room -- the 23 autoregressive baselines the three
tables report come to several hundred gigabytes in float32, and the registry defines ten
more -- and `HF_HUB_OFFLINE=1` once they are cached, so a run uses exactly the cached
weights.

## Running at scale

Each run writes one JSON per example into `<run>/res/NNNN.json` behind an atomic
`O_EXCL` claim file. Two consequences worth using:

* a run resumes where it stopped, so an interrupted job is restarted rather than
  redone;
* several processes can drain the same run at once, on the same node or different
  ones, without coordination.

If a process is stopped in the middle of an example, release that example's claim
before restarting: `nashlib/workqueue.py` has the three-line loop for it.

Examples are handed out longest first. A 231-slot CLAPNQ question costs 138 times what
a 19-slot one does in forward passes -- 28,906 against 209 -- and about 360 times in
seconds, so starting the long ones last would leave a single GPU grinding alone for
minutes after the others had finished.
