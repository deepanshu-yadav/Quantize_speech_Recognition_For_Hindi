import os
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from librosa import resample
from typing import Tuple

class StandaloneASRRNNT:
    def __init__(self, model_dir: str = "model_components"):
        """Initialize the transducer-based ASR system."""
        self.model_dir = model_dir
        self.device = "cuda" if ort.get_device() == "GPU" else "cpu"

        # Ensure model_dir exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")

        # Load ONNX models
        encoder_path = os.path.join(model_dir, "encoder_fp16.onnx")
        decoder_path = os.path.join(model_dir, "decoder_fp16.onnx")
        joiner_path = os.path.join(model_dir, "joiner_fp16.onnx")
        for path in [encoder_path, decoder_path, joiner_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file {path} not found")

        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        providers = ["CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(encoder_path, sess_options=session_opts, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, sess_options=session_opts, providers=providers)
        self.joiner = ort.InferenceSession(joiner_path, sess_options=session_opts, providers=providers)

        # Load encoder metadata
        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.normalize_type = meta.get("normalize_type", "")
        self.pred_rnn_layers = int(meta.get("pred_rnn_layers", 2))
        self.pred_hidden = int(meta.get("pred_hidden", 512))

        # Load vocabulary
        vocab_path = os.path.join(model_dir, "tokens.txt")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file {vocab_path} not found")
        self.id2token = {}
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                t, idx = line.strip().split()
                self.id2token[int(idx)] = t
        self.blank_id = 256  # No_of_vocab - 1  = 257 - 1

    def get_decoder_state(self):
        """Initialize decoder states."""
        batch_size = 1
        state0 = np.zeros((self.pred_rnn_layers, batch_size, self.pred_hidden), dtype=np.float16)
        state1 = np.zeros((self.pred_rnn_layers, batch_size, self.pred_hidden), dtype=np.float16)
        return state0, state1

    def calculate_audio_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute mel-spectrogram features."""
        if sr != 16000:
            audio = resample(audio, orig_sr=sr, target_sr=16000)

        audio = torch.from_numpy(audio).float()
        mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=0.0,
            f_max=8000.0,
            n_mels=80,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False
        )
        audio = audio.unsqueeze(0)
        features = mel_transform(audio)
        features = torch.log(features + 1e-9)

        if self.normalize_type == "per_feature":
            mean = features.mean(dim=2, keepdims=True)
            stddev = features.std(dim=2, keepdims=True) + 1e-5
            features = (features - mean) / stddev

        features = features.squeeze(0).permute(1, 0).numpy().astype(np.float16)  # [T, n_mels]
        return features

    def preprocess_audio_file(self, wav_path: str) -> np.ndarray:
        """Preprocess WAV file to mel-spectrogram features."""
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file {wav_path} not found")

        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.concatenate([audio, np.zeros(sr * 2)])  # Add 2s padding
        return audio

    def run_encoder(self, features: np.ndarray) -> np.ndarray:
        """Run encoder on features."""
        features = torch.from_numpy(features).unsqueeze(0).permute(0, 2, 1)  # [1, n_mels, T]
        x_lens = np.array([features.shape[2]], dtype=np.int64)

        encoder_out, _ = self.encoder.run(
            [self.encoder.get_outputs()[0].name, self.encoder.get_outputs()[1].name],
            {
                self.encoder.get_inputs()[0].name: features.numpy(),
                self.encoder.get_inputs()[1].name: x_lens,
            },
        )
        return encoder_out

    def run_decoder(self, token: int, state0: np.ndarray, state1: np.ndarray):
        """Run decoder for a single token."""
        target = np.array([[token]], dtype=np.int32)
        target_len = np.array([1], dtype=np.int32)

        decoder_out, _, state0_next, state1_next = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
                self.decoder.get_outputs()[1].name,
                self.decoder.get_outputs()[2].name,
                self.decoder.get_outputs()[3].name,
            ],
            {
                self.decoder.get_inputs()[0].name: target,
                self.decoder.get_inputs()[1].name: target_len,
                self.decoder.get_inputs()[2].name: state0,
                self.decoder.get_inputs()[3].name: state1,
            },
        )
        return decoder_out, state0_next, state1_next

    def run_joiner(self, encoder_out: np.ndarray, decoder_out: np.ndarray) -> np.ndarray:
        """Run joiner to combine encoder and decoder outputs."""
        logit = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out,
                self.joiner.get_inputs()[1].name: decoder_out,
            },
        )[0]
        return logit

    def transcribe_file(self, wav_path: str) -> str:
        """Full transcription pipeline for transducer model."""
        try:
            # Preprocess audio
            audio = self.preprocess_audio_file(wav_path)
            return self.transcribe(audio)
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def transcribe(self, audio: np.ndarray) -> str:
        """Full transcription pipeline for transducer model."""
        try:
            # Preprocess audio
            features = self.calculate_audio_features(audio, 16000)
            print(f"Features shape: {features.shape}")

            # Initialize decoding
            ans = [self.blank_id]
            state0, state1 = self.get_decoder_state()
            decoder_out, state0_next, state1_next = self.run_decoder(ans[-1], state0, state1)

            # Run encoder
            encoder_out = self.run_encoder(features)
            # print(f"Decoder dimension {decoder_out.shape}")
            # Decode time steps
            for t in range(encoder_out.shape[2]):
                encoder_out_t = encoder_out[:, :, t : t + 1]
                # print(f"Encoder input to jointnet shape {encoder_out_t.shape}")
                # print("="*20)
                # print(f"Original Encoder input to jointnet shape {encoder_out.shape}")
                logits = self.run_joiner(encoder_out_t, decoder_out)
                logits = torch.from_numpy(logits).squeeze()
                idx = torch.argmax(logits, dim=-1).item()
                if idx != self.blank_id:
                    ans.append(idx)
                    state0 = state0_next
                    state1 = state1_next
                    decoder_out, state0_next, state1_next = self.run_decoder(ans[-1], state0, state1)

            # Convert to text
            ans = ans[1:]  # Remove initial blank
            # print(ans[1:])
            tokens = [self.id2token[i+1536] for i in ans]
            text = "".join(tokens).replace("<unk>", "").replace("▁", " ").strip()
            return text
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")