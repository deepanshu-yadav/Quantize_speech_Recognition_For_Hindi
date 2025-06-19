import os
import json
import pickle
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from librosa import resample
from typing import Tuple
import sentencepiece as spm

class StandaloneASR:
    def __init__(self, model_dir: str = "model_components"):
        """Initialize the ASR system from saved components."""
        self.model_dir = model_dir
        self.device = "cuda" if ort.get_device() == "GPU" else "cpu"

        # Ensure model_dir exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")

        # Load ONNX model
        onnx_path = os.path.join(model_dir, "indicconformer_stt_hi_ctc_only_fp16.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file {onnx_path} not found")
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"]
        )

        # Load decoder config
        decoder_config_path = os.path.join(model_dir, "decoder_config.json")
        if not os.path.exists(decoder_config_path):
            raise FileNotFoundError(f"Decoder config {decoder_config_path} not found")
        with open(decoder_config_path, "r") as f:
            self.decoder_config = json.load(f)

    def _load_tokenizer(self):
        """Load the pickled tokenizer (SentencePiece or MultilingualTokenizer)."""

        sp = spm.SentencePieceProcessor()
        sp.load(os.path.join(self.model_dir, "tokenizer_hi.model"))
        return sp

    def calculate_audio_features(self, audio: np.ndarray, sr: int) -> \
        Tuple[np.ndarray, np.ndarray]:
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = resample(audio, orig_sr=sr, target_sr=16000)

        # Convert to torch tensor
        audio = torch.from_numpy(audio).float()

        # Compute mel-spectrogram
        # Parameters to match NeMo's typical Conformer settings
        mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=0.0,
            f_max=8000.0,
            n_mels=80,  # Typical for Conformer models
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False
        )

        # Add batch dimension and compute mel-spectrogram
        audio = audio.unsqueeze(0)  # [1, time_steps]
        mel_spec = mel_transform(audio)  # [1, n_mels, time_steps]

        # Log mel-spectrogram (common in ASR models)
        mel_spec = torch.log(mel_spec + 1e-9)

        # Normalize (optional, but common in NeMo preprocessing)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

        # Ensure shape is [batch_size, n_mels, time_steps] (e.g., [1, 80, T])
        # No transpose needed since MelSpectrogram outputs [batch_size, n_mels, time_steps]

        # Convert to numpy and FP16
        audio_features = mel_spec.numpy().astype(np.float16)

        # Compute audio length (number of time steps in mel-spectrogram)
        audio_length = np.array([mel_spec.shape[2]], dtype=np.int64)

        # Validate shape
        expected_shape = (1, 80, audio_features.shape[2])
        if audio_features.shape != expected_shape:
            raise ValueError(f"Expected audio features shape {expected_shape}, got {audio_features.shape}")

        return audio_features, audio_length
    
    def preprocess_audio_file(self, wav_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess WAV file to its numppy array."""
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file {wav_path} not found")

        # Read audio
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        audio = audio.astype(np.float32)
        return audio, sr

    def run_inference(self, audio_features: np.ndarray, audio_length: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        input_names = [inp.name for inp in self.session.get_inputs()]
        if len(input_names) != 2:
            raise ValueError(f"Expected 2 inputs, got {len(input_names)}: {input_names}")

        # Ensure input names match expected
        expected_input_name = "audio_signal"
        if input_names[0] != expected_input_name:
            print(f"Warning: First input name is {input_names[0]}, expected {expected_input_name}")

        inputs = {
            input_names[0]: audio_features,
            input_names[1]: audio_length
        }

        try:
            outputs = self.session.run(None, inputs)
            return outputs[0]  # Assuming logits are the first output
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {str(e)}")

    def decode_output(self, logits: np.ndarray) -> str:
        """Decode CTC output using greedy decoding."""
        # Convert logits to numpy if needed
        if isinstance(logits, torch.Tensor):
            logits = logits.numpy()

        # Get predictions (greedy decoding)
        predictions = np.argmax(logits, axis=-1).squeeze(0)

        # CTC decoding: remove blanks and repeats
        blank_id = self.decoder_config.get("blank_idx", logits.shape[-1] - 1)
        decoded = []
        previous = blank_id
        for p in predictions:
            if p != blank_id and p != previous:
                decoded.append(int(p))
            previous = p

        # Convert token IDs to text
        try:
            
            vocab_path = os.path.join(self.model_dir, "tokenizer_hi.vocab")
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Vocab file {vocab_path} not found")

            # Load the vocabulary from the text file
            vocab = {}
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Split each line by tab and take the first column (token)
                    token = line.strip().split('\t')[0]
                    id_hi = int(line.strip().split('\t')[1].replace("-", "")) + 1537
                    vocab[id_hi] = token
            # print(vocab)
            # print(decoded)
            # Decode the sequence of IDs into text
            text = ''.join([vocab[id] if  id in vocab else '<UNK>' for id in decoded])
             # Replace SentencePiece underscore with space
            text = text.replace('▁', ' ').strip()
            return text
           
        except Exception as e:
            raise RuntimeError(f"Decoding failed: {str(e)}")
    
    def transcribe_file(self, wav_path: str) -> str:
        """Transcribe a single WAV file."""
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file {wav_path} not found")
        
        # Preprocess audio and run transcription
        audio, sr = self.preprocess_audio_file(wav_path)
        return self.transcribe(audio, sr)
        
    
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """Full transcription pipeline."""
        try:
            # 1. Preprocess audio
            audio_features, audio_length = self.calculate_audio_features(audio, sr)
            print(f"Audio features shape: {audio_features.shape}, Length: {audio_length}")

            # 2. Run inference
            logits = self.run_inference(audio_features, audio_length)

            # 3. Decode output
            text = self.decode_output(logits)

            return text
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")
