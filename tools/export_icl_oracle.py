#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/export_icl_oracle.py - Export official Qwen3-TTS ICL prompt & reference codes golden data.

Uses .venv-tts and local Qwen3-TTS models.
Extracts:
  - 24kHz mono preprocessed waveform info
  - ECAPA-TDNN speaker embedding
  - 16 groups of Mimi reference codes [frames, 16]
  - Mimi encoder intermediate tensors (SEANet, Transformer, Downsample, RVQ) -> saved to tmp/
  - ICL prefill embedding, first talker logits/token, 15 predictor codes
"""

import os
import sys
import json
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch

from qwen_tts import Qwen3TTSModel


def get_ref_text_pairs(repo_root):
    return [
        {
            "audio_file": "ref_test.wav",
            "audio_path": os.path.join(repo_root, "ref_test.wav"),
            "ref_text": "这是一个错事。",
            "target_text": "你好，世界。",
            "language": "Chinese",
        },
        {
            "audio_file": "This is an English test.wav",
            "audio_path": os.path.join(repo_root, "This is an English test.wav"),
            "ref_text": "This is an English test.",
            "target_text": "Hello world, this is a test.",
            "language": "English",
        }
    ]


def load_and_preprocess_audio_24k(audio_path):
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    if sr != 24000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
    return wav.astype(np.float32)


def run_mimi_encoder_stages(enc_model, wav_24k):
    """
    Step through Mimi encoder stages:
      1. SEANet conv layers -> [1, 512, L_seanet]
      2. Transformer 8 layers -> [1, L_seanet, 512]
      3. Downsample conv -> [1, 512, L_frames]
      4. Split RVQ -> codes [16, frames]
    """
    x = torch.from_numpy(wav_24k).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        seanet_out = enc_model.encoder(x)
        tfm_in = seanet_out.transpose(1, 2)
        tfm_out = enc_model.encoder_transformer(tfm_in)[0]
        down_in = tfm_out.transpose(1, 2)
        down_out = enc_model.downsample(down_in)
        # RVQ 32 quantizers
        codes_all = enc_model.quantizer.encode(down_out, 32)
        # Select first 16 quantizers, transpose to [frames, 16]
        codes_16 = codes_all[:16, 0].transpose(0, 1)

    return {
        "seanet_out": seanet_out.cpu().numpy(),
        "tfm_out": tfm_out.cpu().numpy(),
        "down_out": down_out.cpu().numpy(),
        "codes": codes_16.cpu().numpy(),
    }


def extract_icl_talker_prefill(tts, ref_codes_16, spk_emb, ref_text, target_text, language):
    """
    Extract Talker prefill embeddings, first Talker token and 15 predictor codes.
    """
    model = tts.model
    # 1. Format inputs
    input_text = tts._build_assistant_text(target_text)
    input_ids = tts._tokenize_texts([input_text])
    ref_ids = [tts._tokenize_texts([tts._build_ref_text(ref_text)])[0]]
    
    # 2. Build voice_clone_prompt dict
    voice_clone_prompt = {
        "ref_code": [torch.from_numpy(ref_codes_16)],
        "ref_spk_embedding": [torch.from_numpy(spk_emb)],
        "x_vector_only_mode": [False],
        "icl_mode": [True],
    }
    
    # Talker forward to get prefill embedding
    with torch.no_grad():
        # Talker input embed construction
        # Follow modeling_qwen3_tts.py generate_icl_prompt
        spk_embed = torch.from_numpy(spk_emb).unsqueeze(0).float()
        
        # Talker codec prefill list
        codec_prefill_list = [[
            model.config.talker_config.codec_nothink_id if language.lower() == "auto" else model.config.talker_config.codec_think_id,
            model.config.talker_config.codec_think_bos_id,
            model.config.talker_config.codec_think_eos_id if language.lower() == "auto" else model.config.talker_config.codec_language_id[language.lower()],
        ]]
        if language.lower() != "auto":
            codec_prefill_list[0].append(model.config.talker_config.codec_think_eos_id)
            
        # Call generate_icl_prompt
        text_id = input_ids[0][:, 3:-5]
        ref_id = ref_ids[0][:, 3:-2]
        ref_code_t = torch.from_numpy(ref_codes_16)
        
        tts_bos_embed, tts_eos_embed, tts_pad_embed = model.talker.text_projection(
            model.talker.get_text_embeddings()(
                torch.tensor([[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
                             dtype=torch.long)
            )
        ).chunk(3, dim=1)
        
        icl_input_embed, trailing_text = model.generate_icl_prompt(
            text_id=text_id,
            ref_id=ref_id,
            ref_code=ref_code_t,
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=False
        )

        # Run greedy generate for a few steps to get first frames
        talker_kwargs = {
            "max_new_tokens": 10,
            "min_new_tokens": 2,
            "do_sample": False,
            "subtalker_dosample": False,
            "eos_token_id": model.config.talker_config.codec_eos_token_id,
        }
        gen_codes_list, _ = model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=[language],
            non_streaming_mode=False,
            **talker_kwargs
        )
        first_frame_codes = gen_codes_list[0][0].cpu().numpy().tolist()

    return {
        "icl_input_embed_shape": list(icl_input_embed.shape),
        "icl_input_embed_stats": {
            "mean": float(icl_input_embed.mean().item()),
            "std": float(icl_input_embed.std().item()),
            "first_5": icl_input_embed[0, 0, :5].cpu().numpy().tolist(),
        },
        "first_frame_codes": first_frame_codes,
    }


def main():
    parser = argparse.ArgumentParser(description="Export official Qwen3-TTS ICL Oracle Data")
    parser.add_argument("--update", action="store_true", help="Update golden JSON file")
    parser.add_argument("--model-path", default="models/Qwen3-TTS-12Hz-0.6B-Base", help="Model directory")
    parser.add_argument("--output-json", default="tests/audio/data/ref_codes_golden.json", help="Golden JSON path")
    parser.add_argument("--dump-tmp", action="store_true", default=True, help="Dump intermediate tensors to tmp/")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.abspath(os.path.join(repo_root, args.model_path))
    output_json = os.path.abspath(os.path.join(repo_root, args.output_json))
    tmp_dir = os.path.join(repo_root, "tmp", "icl_oracle")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"[Oracle] Loading Qwen3TTSModel from {model_path}...")
    tts = Qwen3TTSModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32)
    enc_model = tts.model.speech_tokenizer.model.encoder

    ref_pairs = get_ref_text_pairs(repo_root)
    results = {}

    for pair in ref_pairs:
        audio_name = pair["audio_file"]
        audio_path = pair["audio_path"]
        print(f"\n[Oracle] Processing {audio_name} ({pair['language']})...")
        
        if not os.path.exists(audio_path):
            print(f"Error: {audio_path} not found!")
            sys.exit(1)

        wav_24k = load_and_preprocess_audio_24k(audio_path)
        print(f"  24k mono samples: {len(wav_24k)} ({len(wav_24k)/24000.0:.2f}s)")

        # 1. Speaker embedding
        spk_emb = tts.model.extract_speaker_embedding(wav_24k, sr=24000)
        spk_emb_np = spk_emb.cpu().numpy()
        print(f"  Speaker embedding: shape={spk_emb_np.shape}, norm={np.linalg.norm(spk_emb_np):.4f}")

        # 2. Mimi encoder stages
        mimi_res = run_mimi_encoder_stages(enc_model, wav_24k)
        codes = mimi_res["codes"]
        frames, codebooks = codes.shape
        print(f"  Mimi reference codes: shape={codes.shape}")

        if args.dump_tmp:
            np.save(os.path.join(tmp_dir, f"{audio_name}.wav24k.npy"), wav_24k)
            np.save(os.path.join(tmp_dir, f"{audio_name}.seanet.npy"), mimi_res["seanet_out"])
            np.save(os.path.join(tmp_dir, f"{audio_name}.tfm.npy"), mimi_res["tfm_out"])
            np.save(os.path.join(tmp_dir, f"{audio_name}.down.npy"), mimi_res["down_out"])
            np.save(os.path.join(tmp_dir, f"{audio_name}.codes.npy"), codes)

        # 3. Talker ICL prefill & first frame codes
        icl_info = extract_icl_talker_prefill(
            tts, codes, spk_emb_np, pair["ref_text"], pair["target_text"], pair["language"]
        )
        print(f"  First frame codes (Talker + 15 predictors): {icl_info['first_frame_codes']}")

        results[audio_name] = {
            "audio_file": audio_name,
            "language": pair["language"],
            "sample_rate": 24000,
            "samples_24k": len(wav_24k),
            "duration_sec": round(len(wav_24k) / 24000.0, 4),
            "ref_text": pair["ref_text"],
            "target_text": pair["target_text"],
            "speaker_embedding_dim": int(spk_emb_np.shape[0]),
            "speaker_embedding_norm": float(np.linalg.norm(spk_emb_np)),
            "speaker_embedding_first_5": spk_emb_np[:5].tolist(),
            "frames": int(frames),
            "codebooks": int(codebooks),
            "codes": codes.tolist(),
            "icl_info": icl_info,
        }

    if args.update:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Oracle] Successfully updated golden data at: {output_json}")
    else:
        if os.path.exists(output_json):
            with open(output_json, "r", encoding="utf-8") as f:
                golden = json.load(f)
            mismatch = 0
            for name, data in results.items():
                if name not in golden:
                    print(f"[Check] Warning: {name} not in golden data!")
                    mismatch += 1
                    continue
                g_codes = np.array(golden[name]["codes"])
                c_codes = np.array(data["codes"])
                if not np.array_equal(g_codes, c_codes):
                    print(f"[Check] FAILED: Codes mismatch for {name}")
                    mismatch += 1
                else:
                    print(f"[Check] PASSED: Codes matched perfectly for {name}")
            if mismatch == 0:
                print("\n[Check] All golden checks passed!")
            else:
                print(f"\n[Check] {mismatch} checks failed!")
                sys.exit(1)
        else:
            print(f"[Check] Golden file {output_json} does not exist. Run with --update to create it.")


if __name__ == "__main__":
    main()
