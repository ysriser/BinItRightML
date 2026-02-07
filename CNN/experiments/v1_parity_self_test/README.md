# V1 Inference Parity Self-Test

This tool checks **preprocessing + ONNX inference parity** across environments
(Python vs Android). It generates a golden snapshot and compares later runs to
make drift obvious.

## What it Detects
- Top-1 label mismatches
- Probability vector drift (L¡Þ / max abs diff)
- Top-3 overlap drop
- Preprocess mismatches (resize, crop, normalization)

## Quick Start (Beginner)

### 1) Generate a golden snapshot
```
python CNN/experiments/v1_parity_self_test/parity_cli.py \
  --mode golden \
  --images CNN/experiments/v1_parity_self_test/samples
```

If you have a manifest CSV:
```
python CNN/experiments/v1_parity_self_test/parity_cli.py \
  --mode golden \
  --manifest <path_to_manifest.csv>
```

Outputs:
- `CNN/experiments/v1_parity_self_test/outputs/parity_golden.json`
- `CNN/experiments/v1_parity_self_test/outputs/parity_golden.csv`

### 2) Re-run later and compare
First create a new run file (can be on another machine):
```
python CNN/experiments/v1_parity_self_test/parity_cli.py \
  --mode golden \
  --images CNN/experiments/v1_parity_self_test/samples \
  --output-dir CNN/experiments/v1_parity_self_test/outputs/current
```

Then compare:
```
python CNN/experiments/v1_parity_self_test/parity_cli.py \
  --mode compare \
  --golden CNN/experiments/v1_parity_self_test/outputs/parity_golden.json \
  --current CNN/experiments/v1_parity_self_test/outputs/current/parity_golden.json
```

### 3) Android tensor parity (same tensor ¡ú same output)
On Android, export **input tensor** and **probs** to cache. Then:
```
python CNN/experiments/v1_parity_self_test/parity_cli.py \
  --mode android_compare \
  --android-tensor <path_to_input_tensor.bin> \
  --android-probs <path_to_probs.json>
```

If the tensor is identical, Python and Android probs should match closely.

## Kotlin Snippet (Android Export)
```kotlin
// Save input tensor (FloatArray) and probs to app cache.
fun exportParityArtifacts(
    context: Context,
    inputTensor: FloatArray,
    probs: FloatArray
) {
    val cacheDir = context.cacheDir
    val tensorFile = File(cacheDir, "input_tensor.bin")
    val probsFile = File(cacheDir, "probs.json")

    // Write tensor as little-endian float32
    tensorFile.outputStream().use { out ->
        val buffer = ByteBuffer.allocate(inputTensor.size * 4)
            .order(ByteOrder.LITTLE_ENDIAN)
        for (v in inputTensor) buffer.putFloat(v)
        out.write(buffer.array())
    }

    // Write probs JSON
    val probsList = probs.toList()
    probsFile.writeText(Gson().toJson(probsList))
}
```

## Troubleshooting
- **Top-1 mismatch > 0%**: preprocessing drift or label-map mismatch.
- **Prob diff > 0.02** with same input tensor: runtime mismatch or
  ONNX version / provider differences.
- **Audit mismatches**: resize/crop or normalization differs from baseline.

## CLI Reference
- Generate golden:
  `python ... --mode golden --images <dir>`
- Compare:
  `python ... --mode compare --golden <json> --current <json>`
- Android compare:
  `python ... --mode android_compare --android-tensor <bin> --android-probs <json>`
