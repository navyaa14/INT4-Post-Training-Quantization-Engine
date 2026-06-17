INT4 Post-Training Quantization Engine

A from-scratch weight-only post-training quantization (PTQ) engine for
transformer nn.Linear layers, supporting INT8 and INT4 with
per-channel, group-wise affine (scale + zero-point) calibration and true
sub-byte weight packing. No bitsandbytes, torch.ao, or other one-click
quantization API is used -- every step (calibration, quantize, pack,
dequantize) is implemented directly in engine/quant.py so the
quality-vs-compression trade-off is fully inspectable and testable.

Built and benchmarked end-to-end on a small GPT-style decoder trained on a
synthetic token stream, so the whole pipeline (train -> quantize -> evaluate)
runs on CPU in under a minute with no external dataset or pretrained-weight
download required.

Why this exists

Most "quantize my model in one line" libraries hide exactly the decisions
that determine whether quantization tanks model quality: how scales are
calibrated, whether zero-points are used, how outlier weights are handled,
and what the real (not just theoretical) memory footprint is once you
account for packing overhead. This project implements those decisions
explicitly, then measures their effect.

What it does


Calibrates per-output-channel, per-group min/max statistics for each
nn.Linear weight matrix (no calibration dataset needed -- this is
weight-only PTQ, the standard "round-to-nearest" baseline used as a
starting point by toolkits like AutoGPTQ before more elaborate
Hessian-based correction).
Quantizes to an unsigned affine grid (scale, zero_point per
group), producing 4-bit or 8-bit integer codes.
Packs INT4 codes two-per-byte into a uint8 buffer, so the stored
tensor is genuinely 4 bits/weight rather than 8-bit values that happen to
fit in 0-15.
Dequantizes on the fly at forward() time and runs a normal
F.linear, so a QuantLinear is a drop-in replacement for nn.Linear.
Reports validation perplexity, real packed-weight byte counts, and
forward-pass latency, for FP32 / INT8 / INT4, on the same model and the
same evaluation batches.


Project layout

engine/
  model.py       tiny GPT-style decoder (target model for quantization)
  data.py        synthetic structured token stream (train/val split)
  quant.py       the quantization engine itself
  benchmark.py   train -> quantize -> evaluate -> report
tests/
  test_quant.py  unit tests: pack/unpack correctness, error bounds,
                 outlier sensitivity, model-level integration

Usage

bashpip install -r requirements.txt

# train a tiny model and benchmark FP32 vs INT8 vs INT4
python -m engine.benchmark --cpu

# run the test suite
pytest tests/ -v

Quantizing your own model (any nn.Module that uses nn.Linear) is three
lines:

pythonfrom engine.quant import quantize_model

quantize_model(model, bits=4, group_size=32, exclude=["head"])  # in place

exclude keeps any layer whose name contains one of the given substrings in
full precision -- typically the output head, since it sits directly between
the model and the loss.

Results

813K-parameter, 4-layer / 4-head / 128-dim GPT trained for 400 steps on a
synthetic Markov-chain token stream (64-token vocabulary, 64-token context),
group size 32, output head kept in FP32:

precisionval perplexitylinear-layer sizecompression vs FP32latency (ms/fwd)FP3231.0783072.0 KB1.00x46.5INT831.078960.0 KB3.20x49.5INT431.083576.0 KB5.33x52.7

Show Image

INT4 gives a 5.3x reduction in linear-layer storage for a 0.02%
perplexity change on this model/dataset. The compression ratio is below the
theoretical 8x because each group carries a scale and zero_point (stored
as fp32 for simplicity) -- at group_size=32 that overhead is amortized
over 32 weights per group; larger groups shrink the overhead further but
increase quantization error (see below).

Per-layer mean absolute dequantization error (INT4) is reported by
benchmark.py and stays in a tight band (~0.0019-0.0033) across every
attention and MLP projection, which is what you'd want to see -- it would
spike on any layer with a problematic outlier-heavy weight distribution.

Latency note

The forward-pass latency numbers show INT4 as slower than FP32, not
faster. This is expected and worth understanding: this engine dequantizes
to FP32 and runs a standard F.linear matmul, so on stock CPU PyTorch the
extra unpack/dequantize step is pure overhead -- there is no fused INT4 GEMM
kernel here. Real inference speedups from low-bit quantization come from
kernels that compute directly on packed integers (e.g. custom CUDA/Triton
kernels, or libraries like bitsandbytes/AutoGPTQ on GPU). This project's
scope is the storage format and the quality-vs-compression study, not a
production inference kernel -- that's the natural "next step" extension.

Group-size sensitivity

tests/test_quant.py includes a synthetic-outlier test confirming the core
reason group-wise (vs per-tensor) quantization matters: a single severe
outlier weight inflates the scale for its whole group, so shrinking
group_size from 128 down to 16 measurably reduces quantization error on a
row containing one outlier -- the damage is contained to fewer neighbouring
weights.

Design decisions


Affine (asymmetric), not symmetric, quantization -- a zero_point
per group lets the integer grid line up with the actual min/max of that
group's weights, rather than wasting range when the distribution isn't
centered at zero.
Group-wise, not per-tensor or per-layer -- bounds the blast radius of
any single outlier weight to one group instead of the whole row.
Biases stay FP32 -- they're a tiny fraction of parameters; quantizing
them buys negligible memory savings for measurable quality risk.
True bit-packing, not "int8 storing 4-bit values" -- pack_uint4 /
unpack_uint4 actually store two weights per byte, so the byte-count
numbers above reflect real compression, not a theoretical one.


Possible extensions


Activation-aware calibration (AWQ-style scaling) instead of plain RTN
Hessian-based error correction (GPTQ-style) for higher-bit-rate quality
A real packed-INT4 CUDA/Triton kernel to realize inference speedups
Mixed-precision policies chosen automatically per layer based on
measured sensitivity (a built-in version of the per-layer error report)

