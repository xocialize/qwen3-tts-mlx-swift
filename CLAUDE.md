# Qwen3TTS

Internalized Qwen3-TTS 0.6B text-to-speech model for MLX-Swift, forked from [qwen3-asr-swift](https://github.com/ivan-digital/qwen3-asr-swift). Provides the core TTS synthesis engine for the RosettaCast dubbing pipeline — both base synthesis and CustomVoice preset speakers.

## Products

| Product | Type | Description |
|---------|------|-------------|
| `Qwen3TTS` | Library | TTS model: Talker, CodePredictor, SpeechTokenizerDecoder, SpeakerEncoder |
| `AudioCommon` | Library | Audio utilities: file loading, WAV writing, tokenizer, HuggingFace downloader, protocols |

## Public API — Qwen3TTS

**`Qwen3TTSModel`** — main entry point:

```swift
// Load model from HuggingFace cache
let model = Qwen3TTSModel(config: .base06B)
try TTSWeightLoader.load(from: modelPath)  // populates model weights

// Standard synthesis
let audio = model.synthesize(text: "Hello world", language: "english")

// CustomVoice preset speakers (requires CustomVoice model variant)
let audio = model.synthesize(
    text: "Hello world",
    language: "english",
    speaker: "serena",
    instruct: "Speak cheerfully."
)

// Voice cloning from reference audio
let audio = model.synthesizeWithVoiceClone(
    text: "Hello world",
    referenceAudio: refSamples,
    referenceSampleRate: 24000,
    language: "english"
)

// Streaming synthesis
let stream = model.synthesizeStream(text: "Long text...", language: "english")
for try await chunk in stream {
    // chunk.samples, chunk.isFinal
}

// Batch synthesis
let audios = model.synthesizeBatch(texts: ["Hello", "World"], language: "english")
```

**Key types:**
- `TTSModelVariant` — `.base` (aufklarer 4-bit) or `.customVoice` (mlx-community 4-bit)
- `Qwen3TTSConfig` — `talker` + `codePredictor` + `speechTokenizerDecoder` configs. Preset: `.base06B`
- `SamplingConfig` — `temperature` (1.0), `topK` (50), `maxTokens` (500), `repetitionPenalty` (1.05)
- `StreamingConfig` — `firstChunkFrames`, `chunkFrames`, `decoderLeftContext`. Presets: `.default`, `.lowLatency`
- `SpeakerConfig` — preset speaker IDs and dialect mappings (CustomVoice models only)
- `CodecTokens` — special token constants (BOS, EOS, language IDs)
- `TextChunker` — splits long text into synthesis-friendly chunks

## Public API — AudioCommon

**Audio I/O:**
- `AudioFileLoader.load(url:targetSampleRate:)` — load any audio format, resample to target rate
- `AudioFileLoader.loadWAV(url:)` — load WAV, returns `(samples, sampleRate)`
- `AudioFileLoader.resample(_:from:to:)` — sample rate conversion
- `WAVWriter.write(samples:sampleRate:to:)` — write Float32 PCM WAV

**Model Infrastructure:**
- `HuggingFaceDownloader.download(modelName:outputDir:)` — fetch models from HF Hub
- `Qwen3Tokenizer` — `load(from:)`, `decode(tokens:)` — text tokenization
- `CommonWeightLoader.loadSafetensors(at:)` — generic safetensors loader
- `TTSWeightLoader.load(from:)` — TTS-specific weight loading

**Protocols (model capability contracts):**
- `SpeechGenerationModel` — `generate(text:language:)`, `generateStream(...)`
- `SpeechRecognitionModel` — `transcribe(audio:sampleRate:language:)`
- `SpeakerEmbeddingModel` — `embed(audio:sampleRate:) -> [Float]`
- `VoiceActivityDetectionModel` — `detectSpeech(audio:sampleRate:) -> [SpeechSegment]`
- `SpeakerDiarizationModel` — `diarize(audio:sampleRate:) -> [DiarizedSegment]`
- `ModelMemoryManageable` — `isLoaded`, `unload()`, `memoryFootprint`

**Data types:**
- `AudioChunk` — streaming output: `samples`, `sampleRate`, `frameIndex`, `isFinal`
- `SpeakerMel.compute(audio:sampleRate:)` — mel spectrogram for speaker encoder

## Architecture

Internalized fork of qwen3-asr-swift with only Qwen3TTS + AudioCommon targets retained. Key internal components:

- **TalkerModel** — 28-layer transformer (4-bit quantized, GQA with 16 heads / 8 KV heads, RoPE, 1024 hidden)
- **CodePredictorModel** — 5-layer transformer for multi-codebook prediction (16 groups)
- **SpeechTokenizerDecoder** — causal Mimi-based decoder: Split RVQ → transformer → upsampling conv layers → 24kHz audio
- **SpeakerEncoder** — ECAPA-TDNN for x-vector speaker embedding (1024-dim)
- **TextChunker** — sentence-boundary chunking for long texts

**Model specs:** 0.6B/1.7B parameters, 4-bit quantized or full-precision (bfloat16), 12 Hz codec frame rate, 24kHz mono PCM output.

**Supported languages:** English, Japanese, Chinese, Korean, German, Spanish, French, Russian, Italian, Portuguese, Beijing Dialect, Sichuan Dialect.

## Dependencies

### External
- [`mlx-swift`](https://github.com/ml-explore/mlx-swift) (0.21.0+) — MLX, MLXNN, MLXFast
- [`swift-transformers`](https://github.com/huggingface/swift-transformers) (1.1.6+) — Hub for HuggingFace model downloads

## Model Weights

| Resource | HuggingFace ID | Notes |
|----------|---------------|-------|
| Base model (4-bit) | `aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit` | Standard synthesis + x-vector cloning |
| CustomVoice model (4-bit) | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` | 9 preset speakers with emotion instructs |
| VoiceDesign 1.7B (full-precision) | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Voice design via natural language description |
| Base 1.7B (full-precision) | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Standard synthesis + voice cloning |
| CustomVoice 1.7B (full-precision) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 preset speakers with emotion instructs |

**Note:** Full-precision (bfloat16) models are recommended over 4-bit quantized models. The 4-bit quantization causes garbled audio and runaway generation on MLX.

Default cache: `~/.cache/huggingface/hub/models--{org}--{model}/`

## Testing

`Qwen3TTSTests` target with placeholder test. Full model validation runs via integration test CLIs (`icl-clone-eval`, `tts-baseline-test`, `tts-cloning-test`, `tts-customvoice-test`).

## Intentional Divergences from Upstream Python

These changes differ from the original Qwen3-TTS Python repo (`Qwen3-TTS-main/`) and match the corrected HF Spaces demo (https://huggingface.co/spaces/Qwen/Qwen3-TTS). **Do NOT revert these to match the original repo.**

### CausalTransposeConv1d symmetric trimming (`SpeechTokenizerDecoder.swift`)
The original Python `CausalTransConvNet` uses right-only trimming (`left_pad=0, right_pad=pad`). The HF Spaces version corrects this to symmetric trimming (`left_pad=ceil(pad), right_pad=pad-left_pad`). Without this fix, zero-padding artifacts from the transposed convolution accumulate and produce faint tonal hum (~67 Hz / ~540 Hz) at the tail of generated audio.

### Decode length uses `> 0` not `> -1` (`SpeechTokenizerDecoder.swift`)
The original Python `decode()` counted valid frames with `(codes[..., 0] > -1).sum()`, which includes zero-valued padding tokens as valid. The HF Spaces version uses `> 0`, excluding padding. This prevents extra decoded frames from becoming tonal noise at the end of audio output.

### Full-precision weight support (`FlexibleLinear.swift`, `Configuration.swift`, `Talker.swift`, `CodePredictor.swift`)
The original fork only supported 4-bit quantized weights. We added `FlexibleLinear`/`FlexibleMLP` wrapper classes that auto-detect quantization from `config.json` and support both 4-bit and full-precision (bfloat16) weights. The 4-bit quantization was found to cause garbled audio and runaway generation on MLX — full-precision models produce clean output matching the Python reference.

### Non-streaming text prefill for VoiceDesign (`Qwen3TTS.swift`)
The Python `generate_voice_design()` defaults to `non_streaming_mode=True` (line 642 of `qwen3_tts_model.py`), which packs ALL text tokens into the prefill alongside `codec_pad` embeddings. Our Swift `buildPrefillEmbeddings()` now supports both modes via a `nonStreamingMode` parameter. VoiceDesign synthesis routes through non-streaming mode; voice cloning uses streaming mode (matching Python defaults). Without this, multi-sentence text cuts to silence after ~4.5s when the trailing text pool exhausts.

### ICL voice cloning uses sequential text-then-codec layout (`Qwen3TTSCloning`)
**CRITICAL:** ICL voice cloning (non-streaming mode) uses a **sequential** layout — all text positions first (with `codec_pad`), then all codec positions (with `tts_pad`). Text and codec NEVER overlap in non-streaming mode. This is distinct from streaming mode where text and codec can overlap. The sequential layout is verified against `modeling_qwen3_tts.py` `generate_icl_prompt()`. See `com.xocialize.qwen3-tts-cloning/CLAUDE.md` for the full architecture rules and diagnostic table.

### Tonal tail trimming (`Qwen3TTS.swift`)
Post-processing step added to `synthesize()`, `synthesizeWithVoiceClone()`, and `synthesizeBatchInternal()` that detects and removes tonal tail artifacts from generated audio. Not present in the Python repo.

**Root cause:** MLX's `scaledDotProductAttention` produces subtly different logit distributions compared to the CUDA `flash-attn3` kernel used by the HuggingFace Spaces demo. This causes the Talker to generate extra codec tokens past the natural end of speech — those tokens decode to narrow-band tonal hum (~1200 Hz). The demo consistently produces clean audio because flash-attn3's different numerics lead to earlier natural EOS emission.

**Comprehensive code review findings (demo vs main repo vs our Swift):**
- Decoder code (CausalConv1d, CausalTransposeConv1d, ConvNeXtBlock, SnakeBeta, transformer) is functionally identical across all three implementations
- Generation parameters (temperature, top_k, top_p, repetition_penalty) are identical
- The only meaningful difference is the attention kernel: demo uses `kernels-community/flash-attn3`, main repo uses `flash_attention_2`, we use MLX SDPA
- Demo uses no EOS logit bias and `max_new_tokens=2048`; our dub engine uses `eosLogitBias=2.5` and `maxTokens=448`
- Metal FlashAttention (Philip Turner) is available for MLX but wouldn't change logit distributions — it's an efficiency optimization

**Detection approach:** `trimTonalTail()` scans audio in 50ms windows, computing RMS energy variance across consecutive windows. Real speech has variable energy; tonal artifacts have very consistent (low-variance) energy. When a low-variance region is detected at the tail, a fade-out is applied and the rest is truncated.

**EOS logit bias tuning (dub engine parameter):**
- The demo uses no bias at all — worth testing `eosLogitBias=0.0` as a first step
- Adding EOS bias shifts the logit distribution away from what the model was trained with, potentially degrading late-sequence codec quality
- If removing bias doesn't fix length, try values 3.0-5.0 (higher = shorter audio, risk of speech cutoff)

### Seed-once-at-start for reproducible sampling (`Sampling.swift`, `Qwen3TTS.swift`)

**Critical fix discovered during debugging:** Per-token RNG seeding causes runaway generation for certain seed values.

**Problem:** The original implementation called `MLXRandom.seed(seed)` inside `sampleToken()` on every token. This creates **identical Gumbel noise** at every sampling step. If that noise pattern doesn't favor the EOS token, it **never will**, causing the model to generate until hitting `maxTokens`.

**Symptoms:**
- Seeded synthesis: 39+ seconds, hits token limit (500-1000+ tokens)
- Unseeded synthesis: ~4 seconds, natural EOS at ~45 tokens
- Pattern: Large 64-bit FNV-1a hash seeds most likely to fail

**Investigation findings:**
1. Not all seeds fail — certain seeds systematically avoid EOS
2. Large 64-bit seeds (>5×10^18) are more prone to the issue
3. Even small seeds can fail due to identical Gumbel patterns

**Solution:** Seed RNG **once at generation start**, not per-token:
```swift
// In Qwen3TTS.swift - start of generateWithCodePredictor(), runStreamingGeneration(), generateBatchWithCodePredictor()
if let seed = sampling.seed {
    seedSamplingRNG(seed)
}

// In Sampling.swift - new function
public func seedSamplingRNG(_ seed: UInt64) {
    MLXRandom.seed(seed)
}
```

**Additional safety:** 32-bit seed truncation in `speakerSeed()` to avoid problematic large seed values:
```swift
return hash & 0xFFFFFFFF  // Truncate FNV-1a hash to 32-bit
```

**Test results after fix:**
- `test_speaker`: 3.48s, 43 tokens ✓ (was 39.76s runaway)
- `protagonist_male`: 3.45s, 43 tokens ✓ (was 39.76s runaway)
- `narrator_female`: 3.02s, 37 tokens ✓ (was 39.76s runaway)

## Known Issues

- **Metal library not found in SPM CLI builds** — `swift run`/`swift test` fail because SPM doesn't compile Metal shaders. Workaround: copy `mlx-swift_Cmlx.bundle` from Xcode DerivedData (see root CLAUDE.md).
- **Swift 5 language mode** — retained for upstream code compatibility with qwen3-asr-swift conventions.

## Platforms

macOS 15+ (Apple Silicon required for MLX GPU compute)

## Pipeline Position

Core TTS model at the center of the synthesis stack:

```
                    ┌─ DubEngineSynthesis (NativeMLXSynthesisEngine)
Qwen3TTSModel ◄────┤
                    ├─ Qwen3TTSCloning (VoiceCloningEngine)
                    └─ Test harnesses (baseline, cloning, customvoice)
```

Consumed by `com.xocialize.qwen3-tts-cloning` for ICL voice cloning, by `DubEngineSynthesis` for pipeline synthesis, and by all TTS test harnesses.

## Related Packages

- [`com.xocialize.qwen3-tts-cloning`](../com.xocialize.qwen3-tts-cloning/) — ICL voice cloning layer
- [`com.xocialize.mimi-codec-encoder`](../com.xocialize.mimi-codec-encoder/) — sibling codec encoder
- [`com.xocialize.dub-engine`](../com.xocialize.dub-engine/) — pipeline consumer (DubEngineSynthesis)
- [anime-studio root](../CLAUDE.md) — project navigator
