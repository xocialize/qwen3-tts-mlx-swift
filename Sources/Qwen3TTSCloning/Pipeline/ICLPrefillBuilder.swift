import MLX
import MLXNN
import Qwen3TTS

/// Builds the dual-stream prefill embedding sequence for ICL voice cloning.
///
/// ## FIX v4: Sequential text-then-codec layout (matches Python exactly)
///
/// ALL previous versions had a fundamental architectural error: they tried to
/// OVERLAY ref codec embeddings at the same positions as ref text embeddings.
/// The actual Python code (`modeling_qwen3_tts.py` lines 1968-2019) uses a
/// completely different layout: text and codec occupy SEQUENTIAL blocks.
///
/// ### What Python `generate_icl_prompt()` actually does (non-streaming):
///
/// ```
/// TEXT BLOCK:  embed(ref_content + target_content) + tts_eos  |  ALL positions + codec_pad
/// CODEC BLOCK: codec_bos + sum_all_codebooks(ref_code)        |  ALL positions + tts_pad
/// ```
///
/// These two blocks are CONCATENATED, not overlaid. The model sees text first
/// (with codec_pad filler), then codec second (with tts_pad filler). At no
/// position does the model see both real text AND real codec simultaneously.
///
/// ### What the full ICL prefill looks like:
///
/// ```
/// [<|im_start|>assistant\n]           ← roleEmbed (from TARGET text, NOT ref_text)
/// [tts_pad*N + tts_bos + codec_pfx]  ← combinedPrefix (standard, with speaker embed)
/// [ref_content + target_content + tts_eos] + codec_pad  ← TEXT BLOCK
/// [codec_bos + ref_codec_summed] + tts_pad              ← CODEC BLOCK
/// ```
///
/// ### Key facts from the Python source:
///
/// 1. `roleEmbed` = `input_id[:, :3]` = `<|im_start|>assistant\n` from the TARGET
///    text. NOT from ref_text. The ref_text ChatML role is NEVER in the prefill.
///
/// 2. `ref_id[:, 3:-2]` strips ChatML wrapping from ref text, leaving bare content.
///    `input_id[:, 3:-5]` strips ChatML wrapping from target text, leaving bare content.
///    These are concatenated: `cat([ref_content, target_content])` — no ChatML tokens
///    between them, no `<|im_start|>`, no `ref_text`, no `<|im_end|>`.
///
/// 3. The text block has `codec_pad` at every position (not ref codec tokens).
///    The codec block has `tts_pad` at every position (not text tokens).
///    Text and codec are in different position ranges.
///
/// 4. `codec_bos` appears at the START of the codec block, NOT at the end of the
///    prefill as a separate finalPos. There is no `tts_pad + codec_bos` finalPos.
enum ICLPrefillBuilder {

    static func buildPrefillEmbeddings(
        targetTextTokens: [Int],
        prompt: VoiceClonePrompt,
        mode: VoiceCloningMode,
        talker: TalkerModel,
        codePredictor: CodePredictorModel,
        hiddenSize: Int
    ) -> (prefillEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {

        switch mode {
        case .icl:
            return buildICLPrefill(
                targetTextTokens: targetTextTokens,
                prompt: prompt,
                talker: talker,
                codePredictor: codePredictor,
                hiddenSize: hiddenSize
            )
        case .xVectorOnly:
            return buildXVectorPrefill(
                targetTextTokens: targetTextTokens,
                prompt: prompt,
                talker: talker,
                hiddenSize: hiddenSize
            )
        }
    }

    // MARK: - ICL Mode (Sequential text-then-codec, matching Python)

    private static func buildICLPrefill(
        targetTextTokens: [Int],
        prompt: VoiceClonePrompt,
        talker: TalkerModel,
        codePredictor: CodePredictorModel,
        hiddenSize: Int
    ) -> (prefillEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {

        // ━━━ STEP 1: Role embed from TARGET text (not ref_text!) ━━━
        //
        // Python line 2177-2178:
        //   _talker_input_embed_role = text_projection(text_embedding(input_id[:, :3]))
        //   → <|im_start|>assistant\n
        //
        // targetTextTokens = [im_start, assistant, \n, ...target..., im_end, \n, im_start, assistant, \n]

        let targetTokenArray = MLXArray(targetTextTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let targetTextEmbeds = talker.embedText(targetTokenArray)  // [1, targetLen, D]
        let roleEmbed = targetTextEmbeds[0..., 0..<3, 0...]  // [1, 3, D]

        let targetLen = targetTextTokens.count

        // ━━━ STEP 2: TTS special embeddings ━━━

        let ttsPadEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsPad)]).expandedDimensions(axis: 0))
        let ttsBosEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsBos)]).expandedDimensions(axis: 0))
        let ttsEosEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsEos)]).expandedDimensions(axis: 0))

        // ━━━ STEP 3: Codec prefix + speaker embedding ━━━
        //
        // Python lines 2142-2172: standard codec prefix with speaker embed injection.
        // Identical to non-ICL synthesis.

        let langId = CodecTokens.languageId(for: prompt.language) ?? CodecTokens.languageId(for: "english")!
        let codecPrefix: [Int32] = [
            Int32(CodecTokens.codecThink), Int32(CodecTokens.codecThinkBos),
            Int32(langId), Int32(CodecTokens.codecThinkEos),
            Int32(CodecTokens.codecPad), Int32(CodecTokens.codecBos),
        ]
        let codecArray = MLXArray(codecPrefix).expandedDimensions(axis: 0)
        var codecEmbeds = talker.embedCodec(codecArray)  // [1, 6, D]

        // Inject speaker embedding at position 4
        let spkEmbedReshaped = prompt.speakerEmbedding.reshaped([1, 1, hiddenSize])
        let cpPart0 = codecEmbeds[0..., 0..<4, 0...]
        let cpPart1 = codecEmbeds[0..., 4..., 0...]
        codecEmbeds = concatenated([cpPart0, spkEmbedReshaped, cpPart1], axis: 1)  // [1, 7, D]

        let codecLen = codecEmbeds.dim(1)  // 7

        // combinedPrefix = (tts_pad*N + tts_bos) + codec_prefix[:-1]
        let padCount = codecLen - 2  // 5
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, hiddenSize])
        let textOverlay = concatenated([padEmbeds, ttsBosEmbed], axis: 1)  // [1, 6, D]
        let codecWithoutLast = codecEmbeds[0..., 0..<(codecLen - 1), 0...]  // [1, 6, D]
        let combinedPrefix = textOverlay + codecWithoutLast  // [1, 6, D]

        // ━━━ STEP 4: Build ICL content (generate_icl_prompt, lines 1968-2013) ━━━
        //
        // Python:
        //   text_embed = text_projection(text_embedding(cat([ref_id, text_id])))
        //   text_embed = cat([text_embed, tts_eos_embed])
        //
        // ref_id = ref_ids[:, 3:-2] → bare ref transcript content
        // text_id = input_id[:, 3:-5] → bare target text content
        //
        // In our Swift code:
        //   prompt.refTokenIds = tokenizer.encode(referenceText) → already bare content
        //   targetContent = targetTextTokens[3..<(len-5)]

        // Build the concatenated text: [ref_content, target_content]
        let refContentTokens = prompt.refTokenIds.map { Int32($0) }
        let targetContentStart = 3
        let targetContentEnd = targetLen - 5
        var targetContentTokens: [Int32] = []
        if targetContentEnd > targetContentStart {
            targetContentTokens = targetTextTokens[targetContentStart..<targetContentEnd].map { Int32($0) }
        }

        let allContentTokens = refContentTokens + targetContentTokens
        let allContentArray = MLXArray(allContentTokens).expandedDimensions(axis: 0)  // [1, T_text]
        let allContentEmbeds = talker.embedText(allContentArray)  // [1, T_text, D]

        // Append tts_eos
        let textBlock = concatenated([allContentEmbeds, ttsEosEmbed], axis: 1)  // [1, T_text+1, D]
        let textBlockLen = textBlock.dim(1)

        // Overlay text block with codec_pad at EVERY position
        // Python line 2003-2010: text_embed + codec_pad (broadcast)
        let codecPadId = MLXArray([Int32(CodecTokens.codecPad)]).expandedDimensions(axis: 0)
        let codecPadEmbed = talker.embedCodec(codecPadId)  // [1, 1, D]
        let codecPadExpanded = broadcast(codecPadEmbed, to: [1, textBlockLen, hiddenSize])
        let textBlockWithCodecPad = textBlock + codecPadExpanded  // [1, T_text+1, D]

        // ━━━ STEP 5: Build codec block ━━━
        //
        // Python lines 1982-1998:
        //   codec_embed = sum all 16 codebook embeddings per frame
        //   codec_embed = cat([codec_bos_embed, codec_embed])
        //
        // Python line 2012:
        //   cat([text_block, codec_embed + tts_pad_embed])

        let refCodecFrames = prompt.refCodecFrames
        let codecBosEmbed = talker.embedCodec(
            MLXArray([Int32(CodecTokens.codecBos)]).expandedDimensions(axis: 0))  // [1, 1, D]

        let codecBlock: MLXArray
        if refCodecFrames > 0 {
            let refCodecSummed = sumAllCodebookEmbeddings(
                refCodes: prompt.normalizedRefCodes,
                talker: talker,
                codePredictor: codePredictor,
                refCodecFrames: refCodecFrames
            )  // [1, T_ref, D]

            // Prepend codec_bos, overlay ALL positions with tts_pad
            let codecEmbedWithBos = concatenated([codecBosEmbed, refCodecSummed], axis: 1)  // [1, T_ref+1, D]
            let ttsPadForCodec = broadcast(ttsPadEmbed, to: [1, codecEmbedWithBos.dim(1), hiddenSize])
            codecBlock = codecEmbedWithBos + ttsPadForCodec  // [1, T_ref+1, D]
        } else {
            // No ref codes — just codec_bos + tts_pad
            codecBlock = codecBosEmbed + ttsPadEmbed  // [1, 1, D]
        }

        let codecBlockLen = codecBlock.dim(1)

        print("[ICLPrefillBuilder] v4 SEQUENTIAL layout:")
        print("  roleEmbed(3) + combinedPrefix(\(combinedPrefix.dim(1))) + textBlock(\(textBlockLen)) + codecBlock(\(codecBlockLen))")
        print("  textBlock = ref_content(\(refContentTokens.count)) + target_content(\(targetContentTokens.count)) + tts_eos(1) | all + codec_pad")
        print("  codecBlock = codec_bos(1) + ref_codec(\(refCodecFrames)) | all + tts_pad")

        // ━━━ STEP 6: Assemble full prefill ━━━
        //
        // Python lines 2186-2197:
        //   talker_input_embed = cat([roleEmbed, combinedPrefix])
        //   talker_input_embed = cat([talker_input_embed, icl_input_embed])
        //
        // No separate finalPos — the codec_bos is already in the codec block.

        let prefillEmbeds = concatenated([roleEmbed, combinedPrefix, textBlockWithCodecPad, codecBlock], axis: 1)

        // trailing_text_hidden = tts_pad (Python line 2013)
        let trailingTextHidden = ttsPadEmbed  // [1, 1, D]

        let prefillLen = prefillEmbeds.dim(1)
        print("[ICLPrefillBuilder] Total prefill: \(prefillLen) positions")

        return (prefillEmbeds, trailingTextHidden, ttsPadEmbed)
    }

    // MARK: - X-Vector-Only Mode (unchanged)

    private static func buildXVectorPrefill(
        targetTextTokens: [Int],
        prompt: VoiceClonePrompt,
        talker: TalkerModel,
        hiddenSize: Int
    ) -> (prefillEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {

        let textTokenArray = MLXArray(targetTextTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let textEmbeds = talker.embedText(textTokenArray)

        let langId = CodecTokens.languageId(for: prompt.language) ?? CodecTokens.languageId(for: "english")!
        let codecPrefix: [Int32] = [
            Int32(CodecTokens.codecThink), Int32(CodecTokens.codecThinkBos),
            Int32(langId), Int32(CodecTokens.codecThinkEos),
            Int32(CodecTokens.codecPad), Int32(CodecTokens.codecBos),
        ]
        let codecArray = MLXArray(codecPrefix).expandedDimensions(axis: 0)
        var codecEmbeds = talker.embedCodec(codecArray)

        let spkEmbedReshaped = prompt.speakerEmbedding.reshaped([1, 1, hiddenSize])
        let part0 = codecEmbeds[0..., 0..<4, 0...]
        let part1 = codecEmbeds[0..., 4..., 0...]
        codecEmbeds = concatenated([part0, spkEmbedReshaped, part1], axis: 1)

        let ttsPadEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsPad)]).expandedDimensions(axis: 0))
        let ttsBosEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsBos)]).expandedDimensions(axis: 0))
        let ttsEosEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsEos)]).expandedDimensions(axis: 0))

        let codecLen = codecEmbeds.dim(1)
        let padCount = codecLen - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, hiddenSize])
        let textOverlay = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        let codecWithoutLast = codecEmbeds[0..., 0..<(codecLen - 1), 0...]
        let combinedPrefix = textOverlay + codecWithoutLast

        let roleEmbed = textEmbeds[0..., 0..<3, 0...]
        let firstTextEmbed = textEmbeds[0..., 3..<4, 0...]
        let lastCodecEmbed = codecEmbeds[0..., (codecLen - 1)..<codecLen, 0...]
        let firstTextPlusCodec = firstTextEmbed + lastCodecEmbed

        let prefillEmbeds = concatenated([roleEmbed, combinedPrefix, firstTextPlusCodec], axis: 1)

        let textLen = targetTextTokens.count
        let trailStart = 4
        let trailEnd = textLen - 5
        let trailingTextHidden: MLXArray
        if trailEnd > trailStart {
            let trailingSlice = textEmbeds[0..., trailStart..<trailEnd, 0...]
            trailingTextHidden = concatenated([trailingSlice, ttsEosEmbed], axis: 1)
        } else {
            trailingTextHidden = ttsEosEmbed
        }

        return (prefillEmbeds, trailingTextHidden, ttsPadEmbed)
    }

    // MARK: - Helpers

    /// Sum all 16 codebook embeddings for each reference frame.
    ///
    /// Matches Python lines 1983-1989:
    ///   for i in range(num_code_groups):
    ///     if i == 0: codec_embed.append(codec_embedding(ref_code[:, :1]))
    ///     else: codec_embed.append(code_predictor.embeddings[i-1](ref_code[:, i:i+1]))
    ///   codec_embed = cat(codec_embed, dim=1).sum(1).unsqueeze(0)
    private static func sumAllCodebookEmbeddings(
        refCodes: MLXArray, talker: TalkerModel,
        codePredictor: CodePredictorModel, refCodecFrames: Int
    ) -> MLXArray {
        let cb0 = refCodes[0, 0...].asType(.int32).expandedDimensions(axis: 0)
        var summed = talker.embedCodec(cb0)
        for i in 0..<15 {
            let cbi = refCodes[i + 1, 0...].asType(.int32).expandedDimensions(axis: 0)
            summed = summed + codePredictor.embedCodecGroup(cbi, groupIndex: i)
        }
        return summed
    }
}
