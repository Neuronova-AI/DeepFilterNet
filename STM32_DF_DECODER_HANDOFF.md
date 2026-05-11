# STM32 DF Decoder Export Handoff

This note summarizes the work done to make the DeepFilterNet2 DF decoder usable with
ST Edge AI Core 4.0 / STM32 AI Studio, what was tested, and what still needs to be
handled on the embedded side.

## Goal

The original `df_dec.onnx` export could not be imported by ST Edge AI Core 4.0 because
the ONNX graph had a batch/time transpose before the GRU:

```text
INTERNAL ERROR: Transpose on batch not supported in layer _df_gru_gru_Transpose_output_0
```

The goal was to generate an STM32-compatible DF decoder export without changing the
trained weights or retraining the model.

## Code Changes

The export logic is in:

```text
DeepFilterNet/df/scripts/export.py
```

The script now exports a separate STM32-compatible decoder:

```text
tmp/export/df_dec_stm32.onnx
```

The STM32 export uses a one-frame/stateful wrapper:

- input `emb`: rank-4 shape `[1, 1, 1, 256]`
- input `c0`: shape `[1, 64, 1, 96]`
- input `h0`: shape `[1, 2, 1, 256]`
- output `coefs`: shape `[1, 1, 96, 10]`
- output `alpha`: shape `[1, 1, 1]`
- output `h1`: shape `[2, 1, 256]`

The wrapper removes the unsupported batch/time transpose before the GRU. ST Edge AI
also had trouble with the native ONNX GRU in this shape, so the GRU math is expanded
manually using the original trained GRU weights.

Important: the final non-destructive version keeps the full trained GRU depth:

```python
stm32_gru_layers = model.df_dec.df_gru.gru.num_layers
```

An earlier test reduced this to one GRU layer to fit internal flash, but that was
rejected because it severely degraded audio quality.

Latest pushed commit:

```text
c780c0d Keep full GRU depth for STM32 export
```

Pushed to:

```text
https://github.com/Neuronova-AI/DeepFilterNet
```

## Export Command

The export was regenerated using the venv in `Documents/GitHub/.venv`:

```powershell
$env:PYTHONPATH='C:\Users\miche\Documents\GitHub\DeepFilterNet\DeepFilterNet'
C:\Users\miche\Documents\GitHub\.venv\Scripts\python.exe `
  DeepFilterNet\df\scripts\export.py `
  tmp\export `
  --model-base-dir DeepFilterNet2 `
  --opset 12
```

Generated files of interest:

```text
tmp/export/df_dec_stm32.onnx
tmp/export/df_dec_stm32_input.npz
tmp/export/df_dec_stm32_output.npz
```

Current full-depth `df_dec_stm32.onnx` size:

```text
3,429,275 bytes
```

## ONNX Graph Check

The generated model was checked with Python:

```python
import onnx

path = "tmp/export/df_dec_stm32.onnx"
model = onnx.load(path)

print("Transpose nodes:")
for node in model.graph.node:
    if node.op_type == "Transpose":
        perm = None
        for attr in node.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
        print(node.name, perm, list(node.input), "->", list(node.output))

print("\nGRU nodes:")
for node in model.graph.node:
    if node.op_type == "GRU":
        print(node.name, list(node.input), "->", list(node.output))
```

Observed result:

```text
Inputs:
  emb: [1, 1, 1, 256]
  c0:  [1, 64, 1, 96]
  h0:  [1, 2, 1, 256]

Outputs:
  coefs: [1, 1, 96, 10]
  alpha: [1, 1, 1]
  h1:    [2, 1, 256]

GRU nodes: 0
Transpose nodes:
  /Transpose perm=[0, 2, 3, 1]
```

There is no `Transpose perm=[1,0,2]` immediately before a GRU. The GRU is manually
expanded, so there is no ONNX `GRU` node.

## ST Edge AI Validation

Validation command:

```powershell
C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe validate `
  --model tmp\export\df_dec_stm32.onnx `
  --batch-size 1 `
  --mode host `
  --optimization balanced `
  --name network `
  --verbosity 1 `
  --c-api st-ai `
  --target stm32h7 `
  --memory-pool C:\Users\miche\.stm32cubeaistudio\workspace\dcdcsc\.ai\run\run-9\.ai\mempools.json `
  --workspace tmp\st_ai_ws_full_validate `
  --output tmp\st_ai_output_full_validate
```

Validation succeeded.

ST Edge AI summary:

```text
params:       830,383 items
weights:      3,321,576 B / 3.17 MiB
activations:  304,168 B / 297.04 KiB
MACC:         1,068,096
```

ST printed these warnings/errors during report generation:

```text
E: _Einsum_1_output_0_reshape_1_to_chlast layer - number of I/O tensor is not coherent: 0/1
E: _Einsum_output_0_reshape_1_to_chlast layer - number of I/O tensor is not coherent: 0/1
```

Despite those messages, validation completed successfully and generated a report.

## Memory-Pool File Used

The tested memory-pool descriptor was:

```text
C:\Users\miche\.stm32cubeaistudio\workspace\dcdcsc\.ai\run\run-9\.ai\mempools.json
```

It describes these activation memory pools:

```text
DTCMRAM: 128 KiB
ITCMRAM: 64 KiB
RAM_D1:  512 KiB
```

The full-depth model fits RAM:

```text
activations: 297.04 KiB
```

The problem is flash/weights:

```text
weights: 3.17 MiB
internal flash available in GUI: 2 MiB
```

## Internal-Flash Compression Experiment

After the external-flash path was validated, an additional experiment was done to see
whether the model could be made small enough for the 2 MiB internal flash budget
without deleting a GRU layer.

The tested approach was low-rank SVD factorization of the manually expanded GRU gate
matrices. The full model has four large GRU matrices:

```text
df_gru.gru.weight_ih_l0: 768 x 256
df_gru.gru.weight_hh_l0: 768 x 256
df_gru.gru.weight_ih_l1: 768 x 256
df_gru.gru.weight_hh_l1: 768 x 256
```

These account for most of the flash usage. Instead of removing a GRU layer, each
`256 x 256` gate matrix was approximated with two lower-rank matrices. This keeps both
GRU layers and the same stateful one-frame interface, but changes the weights by SVD
approximation.

Experimental ONNX files generated:

```text
tmp/export/df_dec_stm32_lowrank_r72.onnx
tmp/export/df_dec_stm32_lowrank_r64.onnx
```

The rank-72 model is the best candidate found so far. It fits inside the 2 MiB weight
budget, but with little margin:

```text
df_dec_stm32_lowrank_r72.onnx
ONNX size:     2,054,386 B
ST weights:   1,945,320 B / 1.86 MiB
RAM:          304,456 B / 297.32 KiB
MACC:         724,032
```

The rank-64 model has more flash margin, but slightly more quality loss:

```text
df_dec_stm32_lowrank_r64.onnx
ONNX size:     1,857,778 B
ST weights:   1,748,712 B / 1.67 MiB
RAM:          304,424 B / 297.29 KiB
MACC:         674,880
```

Both low-rank ONNX files validated successfully with ST Edge AI Core 4.0 using:

```powershell
C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe validate `
  --model tmp\export\df_dec_stm32_lowrank_r72.onnx `
  --batch-size 1 `
  --mode host `
  --optimization balanced `
  --name network `
  --verbosity 1 `
  --c-api st-ai `
  --target stm32h7 `
  --memory-pool C:\Users\miche\.stm32cubeaistudio\workspace\dcdcsc\.ai\run\run-9\.ai\mempools.json `
  --workspace tmp\st_ai_ws_lowrank_r72 `
  --output tmp\st_ai_output_lowrank_r72
```

and similarly for:

```text
tmp/export/df_dec_stm32_lowrank_r64.onnx
```

Validation summaries:

```text
rank 72:
  params:       486,319 items
  weights:      1,945,320 B / 1.86 MiB
  activations:  304,456 B / 297.32 KiB
  MACC:         724,032

rank 64:
  params:       437,167 items
  weights:      1,748,712 B / 1.67 MiB
  activations:  304,424 B / 297.29 KiB
  MACC:         674,880
```

The same non-fatal ST report messages appeared:

```text
E: _Einsum_1_output_0_reshape_1_to_chlast layer - number of I/O tensor is not coherent: 0/1
E: _Einsum_output_0_reshape_1_to_chlast layer - number of I/O tensor is not coherent: 0/1
```

but validation completed.

## ST Generation Commands Tested

### Normal st-ai Generation

```powershell
C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe generate `
  --model tmp\export\df_dec_stm32.onnx `
  --optimization balanced `
  --name network `
  --verbosity 1 `
  --c-api st-ai `
  --target stm32h7 `
  --memory-pool C:\Users\miche\.stm32cubeaistudio\workspace\dcdcsc\.ai\run\run-9\.ai\mempools.json `
  --workspace tmp\st_ai_ws_full_generate_default `
  --output tmp\st_ai_output_full_generate_default
```

This succeeded, but generated a large embedded weights C file:

```text
tmp/st_ai_output_full_generate_default/network_data.c
size: about 9.0 MB as C source
weights: 3.17 MiB
```

This is why STM32 AI Studio reports internal flash overflow.

### st-ai With External Address

This was tested:

```powershell
C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe generate `
  --model tmp\export\df_dec_stm32.onnx `
  --optimization balanced `
  --name network `
  --verbosity 1 `
  --c-api st-ai `
  --target stm32h7 `
  --memory-pool C:\Users\miche\.stm32cubeaistudio\workspace\dcdcsc\.ai\run\run-9\.ai\mempools.json `
  --workspace tmp\st_ai_ws_full_generate_ext `
  --output tmp\st_ai_output_full_generate_ext `
  --binary `
  --address 0x90000000
```

It failed because `--address` is not supported with the `st-ai` C API:

```text
E102(CliArgumentError): --address/--copy-weights-at arguments are not supported with the 'st-ai' c-api.
```

### Legacy API With Fixed External Address

This command succeeded:

```powershell
C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe generate `
  --model tmp\export\df_dec_stm32.onnx `
  --optimization balanced `
  --name network `
  --verbosity 1 `
  --c-api legacy `
  --target stm32h7 `
  --memory-pool C:\Users\miche\.stm32cubeaistudio\workspace\dcdcsc\.ai\run\run-9\.ai\mempools.json `
  --workspace tmp\st_ai_ws_full_generate_legacy_ext `
  --output tmp\st_ai_output_full_generate_legacy_ext `
  --binary `
  --address 0x90000000
```

Generated files:

```text
tmp/st_ai_output_full_generate_legacy_ext/network_data.bin
tmp/st_ai_output_full_generate_legacy_ext/network_data.c
```

`network_data.c` contains:

```c
#define AI_NETWORK_DATA_WEIGHTS_ADDR     (0x90000000)
```

This is useful if the project can use the legacy API and has external QSPI/OSPI memory
mapped at `0x90000000`.

### st-ai Relocatable Split

This is the preferred non-destructive direction for the `st-ai` API.

First, `arm-none-eabi-gcc` must be in `PATH`. On this machine it was found under
STM32CubeIDE:

```powershell
$gcc = Get-ChildItem -Path 'C:\ST\STM32CubeIDE_1.19.0' `
  -Recurse `
  -Filter arm-none-eabi-gcc.exe `
  -ErrorAction SilentlyContinue |
  Select-Object -First 1

$gccDir = Split-Path $gcc.FullName
$env:PATH = "$gccDir;$env:PATH"
```

Then run:

```powershell
C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe generate `
  --model tmp\export\df_dec_stm32.onnx `
  --optimization balanced `
  --name network `
  --verbosity 1 `
  --c-api st-ai `
  --target stm32h7 `
  --memory-pool C:\Users\miche\.stm32cubeaistudio\workspace\dcdcsc\.ai\run\run-9\.ai\mempools.json `
  --workspace tmp\st_ai_ws_full_generate_reloc_with_gcc `
  --output tmp\st_ai_output_full_generate_reloc_with_gcc `
  --relocatable split
```

This succeeded.

Generated files of interest:

```text
tmp/st_ai_output_full_generate_reloc_with_gcc/network_rel.bin
tmp/st_ai_output_full_generate_reloc_with_gcc/network_data.bin
tmp/st_ai_output_full_generate_reloc_with_gcc/ai_reloc_network.c
tmp/st_ai_output_full_generate_reloc_with_gcc/ai_reloc_network.h
```

Sizes:

```text
network_rel.bin:   51,520 B
network_data.bin:  3,321,576 B
```

ST report:

```text
Runtime Loadable Model - Memory Layout
XIP size:          21,648 B
COPY size:         46,756 B
binary file size:  51,520 B
params file size:  3,321,576 B
params size:       3,321,576 B, not included in the binary file
acts size:         304,168 B
```

This keeps the model intact and splits the weights into an external params blob.

## Audio Quality Benchmark

The test sample used:

```text
sample_000011_chunk_02.wav
```

The clean reference added later:

```text
cleaned.wav
```

The original DeepFilterNet2 output:

```text
tmp/benchmark_df2/cleaned_original.wav
```

The full-depth STM32 wrapper output:

```text
tmp/benchmark_df2/cleaned_stm32_full_step_equiv.wav
```

The destructive one-GRU test output:

```text
tmp/benchmark_df2/cleaned_stm32_1gru_step_equiv.wav
```

Metrics against the clean reference:

```text
Noisy input:
  SNR:     10.582 dB
  PESQ WB: 3.097

Original DeepFilterNet2:
  SNR:     38.768 dB
  PESQ WB: 4.630

STM32 full-depth export-equivalent:
  SNR:     21.890 dB
  PESQ WB: 4.630

STM32 one-GRU destructive version:
  SNR:     7.432 dB
  PESQ WB: 3.474
```

Conclusion: reducing the GRU depth makes the model fit better but degrades quality too
much. The full-depth STM32 export keeps PESQ essentially equal to the original model.

Low-rank compression was also benchmarked against the same clean reference:

```text
STM32 low-rank rank 72:
  SNR:     19.475 dB
  PESQ WB: 4.460

STM32 low-rank rank 64:
  SNR:     17.289 dB
  PESQ WB: 4.427
```

Approximate degradation versus the full-depth STM32 export:

```text
rank 72:
  SNR drop:      21.890 -> 19.475 dB, -2.415 dB
  PESQ WB drop:  4.630 -> 4.460, -0.170
  weights:       3.17 MiB -> 1.86 MiB
  MACC:          1,068,096 -> 724,032

rank 64:
  SNR drop:      21.890 -> 17.289 dB, -4.601 dB
  PESQ WB drop:  4.630 -> 4.427, -0.203
  weights:       3.17 MiB -> 1.67 MiB
  MACC:          1,068,096 -> 674,880
```

Current recommendation if internal flash is mandatory:

```text
Use rank 72 first. It is the best quality/size compromise found so far.
Use rank 64 only if the final firmware/linker overhead makes rank 72 too tight.
```

## Conclusion

There are now two viable directions.

### Option A: No Compression, External Flash

This is the highest-quality path:

1. Keep the full trained GRU depth in `df_dec_stm32.onnx`.
2. Use the STM32-compatible one-frame/stateful wrapper to avoid unsupported ONNX
   transposes and GRU import issues.
3. Do not prune GRU layers to fit flash.
4. Store the weights externally.

The RAM footprint is acceptable:

```text
about 297 KiB activations
```

The flash/internal storage footprint is not acceptable for a 2 MiB internal-flash-only
target:

```text
about 3.17 MiB weights
```

Recommended hardware direction:

```text
Add external QSPI/OSPI flash.
Minimum practical size: 4 MB.
Recommended size: 8 MB or more.
```

The `st-ai --relocatable split` flow produces a small model binary plus a separate
weight blob:

```text
network_rel.bin   -> small loadable model binary
network_data.bin  -> 3.17 MiB weights/params blob
```

The embedded engineer should decide how to place or load `network_data.bin` from
external memory and integrate the generated `ai_reloc_network.c/.h` loader path.

### Option B: Low-Rank GRU, Internal Flash

This is the best internal-flash-only path found so far:

```text
tmp/export/df_dec_stm32_lowrank_r72.onnx
```

It should fit under the 2 MiB internal flash budget for AI weights:

```text
weights: about 1.86 MiB
RAM:     about 297 KiB
```

Quality loss on the current sample:

```text
PESQ WB: 4.630 -> 4.460
SNR:     21.890 dB -> 19.475 dB
```

This is much less destructive than the one-GRU-layer version, which dropped to:

```text
PESQ WB: 3.474
SNR:     7.432 dB
```

The low-rank export is still experimental and should be tested on more speech/noise
samples before freezing it for production.
