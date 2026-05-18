# Qwen3-TTS Swift API Reference

## Package Architecture

The TTS system is split into two Swift packages:

| Package | Purpose | Key Types |
|---------|---------|-----------|
| **Qwen3TTS** | Core TTS engine: model loading, standard synthesis, streaming, batch generation | `Qwen3TTSModel`, `SamplingConfig`, `StreamingConfig`, `AudioChunk` |
| **Qwen3TTSCloning** | ICL voice cloning extension: reference encoding, prefill construction, vocoder post-processing | `VoiceCloningEngine`, `VoiceClonePrompt`, `VoiceCloningMode` |

All audio is 24 kHz mono Float32 PCM. One codec frame = 80 ms (12.5 Hz).

---

## 1. Model Loading

### From HuggingFace (auto-download)

```swift
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit",
    tokenizerModelId: "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    progressHandler: { progress, status in
        print("\(Int(progress * 100))%: \(status)")
    }
)
```

### From a local directory

```swift
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: "/path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)
```

The loader auto-detects model type (`base`, `custom_voice`, `voice_design`),
quantization (4-bit vs full-precision), and dimensions from `config.json`.

### Available Model Variants

| Variant | `TTSModelVariant` | Use Case |
|---------|------------------|----------|
| 0.6B Base 4-bit | `.base` | Voice cloning, fastest inference |
| 0.6B CustomVoice 4-bit | `.customVoice` | Preset speakers |
| 1.7B Base 4-bit | `.base17B` | Voice cloning, higher quality |
| 1.7B Base full-precision | `.base17BFP` | Voice cloning, best quality |
| 1.7B CustomVoice 4-bit | `.customVoice17B` | 9 preset speakers + instruction |
| 1.7B VoiceDesign 4-bit | `.voiceDesign17B` | Voice creation from descriptions |
| 1.7B VoiceDesign full-precision | `.voiceDesign17BFP` | Best VoiceDesign quality |

### Manual local loading (without fromPretrained)

```swift
let modelDir = URL(fileURLWithPath: "/path/to/model")
let config = try Qwen3TTSConfig.fromConfigJSON(at: modelDir)
let model = Qwen3TTSModel(config: config)

// Load weights
try TTSWeightLoader.loadTalkerAndCodePredictorWeights(
    talker: model.talker,
    codePredictor: model.codePredictor,
    from: modelDir)

// Speech tokenizer (decoder) — may be in a subdirectory for 1.7B models
let speechTokenizerDir = modelDir.appendingPathComponent("speech_tokenizer")
try TTSWeightLoader.loadSpeechTokenizerDecoderWeights(
    into: model.codecDecoder,
    from: speechTokenizerDir)

// Speaker encoder (Base models only — needed for voice cloning)
try TTSWeightLoader.loadSpeakerEncoderWeights(
    into: model.speakerEncoder,
    from: modelDir)

// Tokenizer
let tokenizer = Qwen3Tokenizer()
try tokenizer.load(from: modelDir.appendingPathComponent("vocab.json"))
model.setTokenizer(tokenizer)

// Warm up compiled Metal shaders (recommended — adds ~300ms, saves on every generation)
model.warmUp()
```

### Memory management

```swift
model.unload()          // Release all model weights
model.memoryFootprint   // Current memory usage in bytes
model.isLoaded          // Whether weights are in memory
```

---

## 2. Standard Synthesis (`Qwen3TTSModel`)

### Basic synthesis

```swift
let audio: [Float] = model.synthesize(
    text: "Hello, how are you today?",
    language: "english"
)
// audio is Float32 PCM at 24 kHz
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `String` | required | Text to synthesize |
| `language` | `String` | `"english"` | Output language (see supported languages below) |
| `speaker` | `String?` | `nil` | Preset speaker name (CustomVoice models only) |
| `instruct` | `String?` | `nil` | Style instruction or voice description |
| `sampling` | `SamplingConfig` | `.default` | Sampling parameters (see §6) |

### Supported languages

`"english"`, `"chinese"`, `"japanese"`, `"korean"`, `"german"`, `"french"`,
`"spanish"`, `"russian"`, `"italian"`, `"portuguese"`, `"beijing_dialect"`,
`"sichuan_dialect"`

Short forms also work: `"en"`, `"zh"`, `"ja"`, `"ko"`, `"de"`, `"fr"`, `"es"`,
`"ru"`, `"it"`, `"pt"`

---

## 3. CustomVoice Synthesis

Requires a CustomVoice model (`Qwen3-TTS-12Hz-{size}-CustomVoice`).

```swift
let audio = model.synthesize(
    text: "I can't believe it's finally happening!",
    language: "english",
    speaker: "vivian",
    instruct: "Speak with excitement and joy."
)
```

### Available speakers (1.7B CustomVoice)

**Female:** `vivian`, `serena`, `aria`, `emma`, `lily`
**Male:** `ryan`, `david`, `jack`, `leo`, `jason`

Query available speakers at runtime:

```swift
model.availableSpeakers  // → ["aria", "david", "emma", ...]
```

### Instruct examples

| Instruction | Effect |
|-------------|--------|
| `"Speak naturally."` | Neutral delivery (auto-applied if omitted) |
| `"Speak cheerfully with excitement."` | Happy, energetic |
| `"Speak slowly and solemnly."` | Deliberate, serious |
| `"Whisper softly."` | Whispered delivery |
| `"Speak angrily."` | Aggressive tone |
| `"用特别愤怒的语气说"` | Angry tone (Chinese instruction) |

---

## 4. VoiceDesign Synthesis

Requires a VoiceDesign model (`Qwen3-TTS-12Hz-{size}-VoiceDesign`).

Creates new voices from natural-language descriptions.

```swift
let audio = model.synthesizeVoiceDesign(
    text: "Welcome to the grand adventure!",
    voiceDescription: "A warm, confident male narrator with a slight British accent, speaking at a moderate pace.",
    language: "english",
    sampling: .voiceDesign  // temp=0.8, topP=0.9 — more expressive
)
```

### Voice description guidelines (7 dimensions)

1. **Gender** — male, female, neutral
2. **Age** — child, young adult, middle-aged, elderly
3. **Pitch** — low, medium, high
4. **Pace** — slow, moderate, fast
5. **Emotion** — calm, cheerful, sad, angry, excited, serious
6. **Vocal characteristics** — raspy, smooth, breathy, nasal, resonant, clear, magnetic
7. **Use case** — narrator, announcer, conversational, dramatic, animation

Recommended description length: 15–40 words in English or Chinese.

### Reproducible voices with seeds

VoiceDesign is stochastic — same description produces different voices each call.
Use a fixed seed for consistent voice timbre:

```swift
var sampling = SamplingConfig.voiceDesign
sampling.seed = SamplingConfig.speakerSeed(from: "protagonist_male")

let audio = model.synthesizeVoiceDesign(
    text: "Some dialogue.",
    voiceDescription: "A calm young male voice...",
    sampling: sampling
)
```

`speakerSeed(from:)` derives a stable 32-bit hash from any string, safe for
repeated use across sessions and projects.

---

## 5. Voice Cloning

### X-Vector Mode (simple, Base model built-in)

Requires a Base model. Uses only the speaker embedding — no transcript needed.
Lower quality (~0.75 similarity) but simplest to use.

```swift
let audio = model.synthesizeWithVoiceClone(
    text: "I won't let you get away with this!",
    referenceAudio: referenceFloatSamples,  // [Float] at any sample rate
    referenceSampleRate: 24000,
    language: "english"
)
```

### Full ICL Mode (highest quality, Qwen3TTSCloning package)

Requires a Base model with codec encoder weights. Uses reference codes + transcript +
speaker embedding. Highest similarity (~0.89).

```swift
import Qwen3TTSCloning

// 1. Create the cloning engine (once per model load)
let engine = VoiceCloningEngine(model: baseModel, tokenizer: tokenizer)

// 2. Create a reusable prompt from reference audio (once per voice)
//    DO NOT pre-pad the audio with silence — ReferenceEncoder handles this internally.
let prompt = try engine.createPrompt(
    referenceAudio: referenceFloatSamples,  // raw samples, no silence padding
    referenceText: "The exact words spoken in the reference audio.",
    sampleRate: 24000,
    language: "english"
)

// 3. Synthesize with the cloned voice (unlimited reuse)
let audio = try engine.synthesize(
    text: "Hello, this is the cloned voice speaking.",
    prompt: prompt,
    mode: .icl
)
```

### Prompt reuse and serialization

```swift
// Save prompt to disk (safetensors + JSON sidecar)
let url = URL(fileURLWithPath: "/path/to/protagonist.safetensors")
try PromptSerializer.save(prompt, to: url)

// Load prompt later (no re-encoding needed)
let loaded = try PromptSerializer.load(from: url)
let audio = try engine.synthesize(text: "New line.", prompt: loaded, mode: .icl)
```

### VoiceDesign-to-Clone Pipeline

The recommended approach for consistent character voices:

1. Generate a reference clip with VoiceDesign (one-time design phase)
2. Feed that audio + its transcript into the Base model's ICL voice cloning
3. All subsequent lines get the same voice timbre from the shared reference

```swift
// Step 1: Generate reference with VoiceDesign model
let refAudio = voiceDesignModel.synthesizeVoiceDesign(
    text: "The ancient kingdom stood at the crossroads of destiny.",
    voiceDescription: "A calm young female voice, clear and sweet.",
    sampling: .voiceDesign
)
// Save refAudio as WAV for inspection

// Step 2: Swap to Base model (unload VoiceDesign first if memory constrained)
let engine = VoiceCloningEngine(model: baseModel, tokenizer: tokenizer)

// Step 3: Create prompt — DO NOT add silence, ReferenceEncoder handles it
let prompt = try engine.createPrompt(
    referenceAudio: refAudio,
    referenceText: "The ancient kingdom stood at the crossroads of destiny.",
    sampleRate: 24000,
    language: "english"
)

// Step 4: Clone for all dialogue lines
for line in dialogueLines {
    let audio = try engine.synthesize(text: line, prompt: prompt, mode: .icl)
    // Save or play audio
}
```

### Reference audio guidelines

| Property | Recommendation |
|----------|---------------|
| **Ideal length** | 10–15 seconds |
| **Minimum** | 3 seconds (the headline "3-second clone") |
| **Maximum** | 30 seconds (longer risks generation hangs) |
| **Quality** | Clean, noise-free, no music/background |
| **Speech content** | ≥60% of duration should be speech |
| **Transcript** | Must be accurate for ICL mode |

### VoiceCloningMode

| Mode | Prefill Tokens | Similarity | Requires Transcript | Speed |
|------|---------------|------------|---------------------|-------|
| `.icl` | ~80+ | ~0.89 | Yes | Slower prefill |
| `.xVectorOnly` | ~10 | ~0.75 | No | Fastest |

---

## 6. SamplingConfig Reference

```swift
var config = SamplingConfig(
    temperature: 0.5,           // 0.0–1.5: creativity vs determinism
    topK: 50,                   // 1–100: limits to top K tokens
    topP: 1.0,                  // 0.0–1.0: nucleus sampling threshold
    repetitionPenalty: 1.05,    // 1.0–2.0: penalizes repeated tokens
    maxTokens: 4096,            // max codec frames to generate
    eosLogitBias: 0.0,          // adjusts stop probability
    seed: nil                   // UInt64? for reproducible output
)
```

### Parameter details

**temperature** — Controls randomness. At 0.0 the output is fully deterministic
(greedy decoding). At 0.5 (default) the output is balanced. At 0.8 (VoiceDesign
default) the output is more expressive and varied. Above 1.0 becomes unstable.

**topK** — After temperature scaling, only the top K most likely tokens are
considered. Lower values make output more predictable. Set to 1 for greedy.

**topP** — Nucleus sampling: considers the smallest set of tokens whose cumulative
probability exceeds topP. Set to 1.0 to disable (use topK only). VoiceDesign
uses 0.9 for more natural variation.

**repetitionPenalty** — Tokens that have appeared before get their logits divided
(positive logits) or multiplied (negative logits) by this factor. At 1.0 there is
no penalty. At 1.05 (default) mild repetition suppression. At 1.5 strong
suppression (used in x-vector voice cloning to prevent stuttering).

**maxTokens** — Safety cap on codec frames generated. Each frame = 80ms audio.
1000 tokens ≈ 80 seconds. 2048 tokens ≈ 164 seconds. The model should hit EOS
naturally before this limit. If it doesn't, the output is truncated.

**eosLogitBias** — Additive bias to the EOS token's logit. Positive values make
the model stop sooner. Negative values make it generate longer. Default 0.0 matches
HuggingFace demo behavior.

**seed** — When set, the MLX random number generator is seeded once at the start
of generation (not per-token). Same seed + same text + same model = same output.
Use `SamplingConfig.speakerSeed(from: "character_name")` to derive stable seeds
from character identifiers.

### Presets

| Preset | Temp | TopK | TopP | RepPenalty | MaxTokens | Use |
|--------|------|------|------|------------|-----------|-----|
| `.default` | 0.5 | 50 | 1.0 | 1.05 | 4096 | Standard synthesis |
| `.greedy` | 0.0 | 1 | 1.0 | 1.05 | 4096 | Deterministic output |
| `.voiceDesign` | 0.8 | 50 | 0.9 | 1.05 | 2048 | Creative voice generation |

---

## 7. Streaming Synthesis

Returns audio chunks as they are generated for low-latency playback.

```swift
let stream = model.synthesizeStream(
    text: "A longer piece of text for streaming output.",
    language: "english",
    streaming: .default  // 240ms first packet, 2s subsequent chunks
)

for try await chunk in stream {
    // chunk.samples: [Float] — PCM audio for this chunk
    // chunk.sampleRate: Int — always 24000
    // chunk.frameIndex: Int — starting frame index of this chunk
    // chunk.isFinal: Bool — true on the last chunk
    // chunk.elapsedTime: Double — seconds since generation started
    playAudio(chunk.samples)
}
```

### StreamingConfig

```swift
let config = StreamingConfig(
    firstChunkFrames: 3,    // frames in first chunk (3 = 240ms)
    chunkFrames: 25,        // frames per subsequent chunk (25 = 2s)
    decoderLeftContext: 10   // overlap frames for decoder quality
)
```

| Preset | First Packet | Chunk Size | Use |
|--------|-------------|------------|-----|
| `.default` | ~240ms (3 frames) | 2s (25 frames) | Balanced |
| `.lowLatency` | ~80ms (1 frame) | 1.2s (15 frames) | Conversational AI |

### VoiceDesign streaming

```swift
let stream = model.synthesizeVoiceDesignStream(
    text: "Long text here.",
    voiceDescription: "A cheerful narrator.",
    language: "english",
    streaming: .lowLatency
)
```

---

## 8. Batch Synthesis

Generate multiple texts in parallel. Items are processed in lockstep — items
that finish early receive padding until all items are done.

```swift
let texts = [
    "First line of dialogue.",
    "Second line, a bit longer.",
    "Third line."
]

let audioArrays: [[Float]] = model.synthesizeBatch(
    texts: texts,
    language: "english",
    maxBatchSize: 4  // max items per batch
)
// audioArrays[i] corresponds to texts[i]
```

Texts are automatically sorted by length internally to minimize padding waste,
then results are returned in the original input order.

Memory: each item uses ~110 MB KV cache per 1000 tokens. B=4 at 1500 tokens ≈ 660 MB.

---

## 9. Long Text with TextChunker

For text longer than ~35 words, split into natural chunks to avoid quality
degradation and generation hangs:

```swift
let chunks = TextChunker.chunk(longText, maxWords: 35)
var allAudio: [Float] = []

for chunk in chunks {
    let audio = model.synthesize(text: chunk, language: "english")
    allAudio.append(contentsOf: audio)
}
```

`TextChunker.chunk()` finds natural break points in this priority order:
sentence endings (`.!?`) → semicolons/colons → commas → conjunctions
("and", "but", "because"...) → phrase starters ("in the", "on the"...) →
hard word-boundary cut.

Minimum chunk size is 8 words to avoid tiny fragments.

---

## 10. Special Token Constants

Available via `CodecTokens`:

| Token | ID | Purpose |
|-------|-----|---------|
| `codecPad` | 2148 | Padding in codec stream |
| `codecBos` | 2149 | Beginning of codec sequence |
| `codecEos` | 2150 | End of generation |
| `codecThink` | 2154 | Think mode prefix |
| `codecThinkBos` | 2156 | Think block start |
| `codecThinkEos` | 2157 | Think block end |
| `ttsPad` | 151671 | Padding in text stream |
| `ttsBos` | 151672 | Beginning of text-to-speech |
| `ttsEos` | 151673 | End of text stream |

Language IDs: `CodecTokens.languageId(for: "english")` → `2050`

---

## 11. Error Handling

### TTSError (Qwen3TTS package)

| Case | When |
|------|------|
| `.tokenizerNotLoaded` | `setTokenizer()` not called before synthesis |
| `.unknownLanguage(String)` | Unrecognized language identifier |

### VoiceCloningError (Qwen3TTSCloning package)

| Case | When |
|------|------|
| `.emptyReferenceAudio` | Reference audio array is empty |
| `.emptyReferenceText` | Reference text transcript is empty |
| `.emptyTargetText` | Target synthesis text is empty |
| `.encoderWeightsNotLoaded` | Codec encoder not loaded (not a Base model?) |
| `.invalidSampleRate(Int)` | Unsupported sample rate |
| `.promptCreationFailed(String)` | Failed to create voice clone prompt |
| `.synthesisFailedEOS` | Generation hit max tokens without natural EOS |
| `.serializationFailed(String)` | Prompt save failed |
| `.deserializationFailed(String)` | Prompt load failed |

---

## 12. Common Patterns

### Save audio as WAV

```swift
import AVFoundation

func saveWAV(audio: [Float], to url: URL) {
    var int16 = audio.map { Int16(max(-1, min(1, $0)) * Float(Int16.max)) }
    var format = AudioStreamBasicDescription(
        mSampleRate: 24000,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
        mBytesPerPacket: 2, mFramesPerPacket: 1,
        mBytesPerFrame: 2, mChannelsPerFrame: 1,
        mBitsPerChannel: 16, mReserved: 0)

    var file: AudioFileID?
    AudioFileCreateWithURL(url as CFURL, kAudioFileWAVEType, &format, .eraseFile, &file)
    var bytes = UInt32(int16.count * 2)
    AudioFileWriteBytes(file!, false, 0, &bytes, &int16)
    AudioFileClose(file!)
}
```

### Consistent character voice across a project

```swift
// Design phase (once)
var designSampling = SamplingConfig.voiceDesign
designSampling.seed = SamplingConfig.speakerSeed(from: "hero_v1")

let refAudio = voiceDesignModel.synthesizeVoiceDesign(
    text: "Reference text spoken naturally for 10-15 seconds.",
    voiceDescription: "A brave young male voice...",
    sampling: designSampling
)

// Clone phase (once per voice)
let prompt = try engine.createPrompt(
    referenceAudio: refAudio,
    referenceText: "Reference text spoken naturally for 10-15 seconds.",
    sampleRate: 24000, language: "english"
)
try PromptSerializer.save(prompt, to: heroPromptURL)

// Production phase (unlimited reuse)
let heroPrompt = try PromptSerializer.load(from: heroPromptURL)
for scene in script.scenes {
    for line in scene.heroLines {
        let audio = try engine.synthesize(text: line, prompt: heroPrompt, mode: .icl)
        exportAudio(audio, for: line)
    }
}
```

### Checking model capabilities

```swift
model.modelType        // "base", "custom_voice", or "voice_design"
model.isVoiceDesign    // true if VoiceDesign model
model.hasCodecEncoder  // true if codec encoder loaded (Base models)
model.availableSpeakers // ["vivian", "ryan", ...] for CustomVoice, [] for others
```
