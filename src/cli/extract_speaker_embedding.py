import os
import sys
import torch
import numpy as np
import soundfile as sf

def main():
    if len(sys.argv) < 4:
        print("Usage: python extract_speaker_embedding.py <model_dir> <audio_path> <output_bin_path>")
        sys.exit(1)
        
    model_dir = sys.argv[1]
    audio_path = sys.argv[2]
    output_bin = sys.argv[3]
    
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    qwen_tts_path = os.path.join(repo_root, "Qwen3-TTS")
    if os.path.exists(qwen_tts_path):
        sys.path.append(qwen_tts_path)
    
    # Import locally
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder, mel_spectrogram
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    except ImportError as e:
        print(f"Error: Failed to import Qwen3-TTS libraries. Ensure {qwen_tts_path} exists and is valid. Detail: {e}")
        sys.exit(1)
        
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
        
    # Load audio
    try:
        wav, sr = sf.read(audio_path)
    except Exception as e:
        print(f"Error: Failed to read audio file: {e}")
        sys.exit(1)
        
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1) # to mono
        
    # Auto-resample if samplerate is not 24000Hz
    if sr != 24000:
        # Simple linear resampling to avoid librosa dependency
        num_samples = int(len(wav) * 24000 / sr)
        wav = np.interp(np.linspace(0, len(wav) - 1, num_samples), np.arange(len(wav)), wav)
        
    # Load model config and speaker_encoder weights
    try:
        config = Qwen3TTSConfig.from_pretrained(model_dir)
        spk_config = config.speaker_encoder_config
        encoder = Qwen3TTSSpeakerEncoder(spk_config)
        
        # Load weights
        from safetensors import safe_open
        state_dict = {}
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("speaker_encoder."):
                    name = k[len("speaker_encoder."):]
                    state_dict[name] = f.get_tensor(k)
        encoder.load_state_dict(state_dict)
        encoder.eval()
    except Exception as e:
        print(f"Error: Failed to load speaker encoder model or weights: {e}")
        sys.exit(1)
        
    # Extract Mel
    mels = mel_spectrogram(
        torch.from_numpy(wav).unsqueeze(0).float(),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000
    ).transpose(1, 2)
    
    # Forward pass
    with torch.no_grad():
        spk_emb = encoder(mels)[0]
        
    # Save to file
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_bin)), exist_ok=True)
        spk_emb_np = spk_emb.numpy().astype(np.float32)
        spk_emb_np.tofile(output_bin)
        print(f"Success: Extracted speaker embedding (dim={spk_emb_np.shape[0]}) saved to {output_bin}")
    except Exception as e:
        print(f"Error: Failed to write output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
