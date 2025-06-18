import torch
import os
import sounddevice as sd
import numpy as np
from queue import Queue
import threading
import sys

# Assume your StandaloneASR class is defined in asr_module.py
# from asr_module import StandaloneASR 
from inference_ctc_float16_non_streaming import StandaloneASR


class StreamingASR:
    def __init__(self, asr_model: StandaloneASR, sample_rate=16000, chunk_size=512, silence_threshold=0.5):
        self.asr_model = asr_model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold_seconds = silence_threshold

        # Load Silero VAD model
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.vad_utils

        self.audio_queue = Queue()
        self.is_running = False

    def _audio_callback(self, indata, frames, time, status):
        """This is called by sounddevice for each new audio chunk."""
        if status:
            print(status)
        self.audio_queue.put(bytes(indata))

    def _process_audio(self):
        """Main processing loop for VAD and ASR."""
        audio_buffer = []
        is_speaking = False
        silent_chunks = 0
        num_chunks_for_silence = int((self.sample_rate * self.silence_threshold_seconds) / self.chunk_size)

        print("Listening... (Speak into the microphone)")

        while self.is_running:
            try:
                # Get audio chunk from the queue
                audio_chunk_bytes = self.audio_queue.get()
                audio_float32 = np.frombuffer(audio_chunk_bytes, dtype=np.float32)
                
                # Convert to torch tensor for VAD
                audio_tensor = torch.from_numpy(audio_float32)

                # Get VAD probability
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

                if speech_prob > 0.5:  # Threshold for speech detection
                    if not is_speaking:
                        print("Speech detected, recording...")
                        is_speaking = True
                    audio_buffer.append(audio_float32)
                    silent_chunks = 0
                else:
                    if is_speaking:
                        silent_chunks += 1
                        if silent_chunks > num_chunks_for_silence:
                            print("Silence detected, transcribing...")
                            is_speaking = False
                            
                            # We have a full utterance, process it
                            full_utterance = np.concatenate(audio_buffer)
                            audio_buffer = []

                            # --- Re-use your existing logic ---
                            # Your ASR model expects features, not raw audio.
                            # So we adapt your preprocess_audio function slightly.
                            
                            # 1. Preprocess audio
                            # We can't use the file-based preprocess_audio directly,
                            # but we can reuse its core logic.
                            try:
                                features, length = self.asr_model.calculate_audio_features(full_utterance, 
                                                                                           self.sample_rate)
                                
                                # 2. Run inference
                                logits = self.asr_model.run_inference(features, length)
                                
                                # 3. Decode output
                                text = self.asr_model.decode_output(logits)
                                
                                print(f"Transcription: {text}\nListening...")
                            except Exception as e:
                                print(f"Error during transcription: {e}")

            except self.audio_queue.Empty:
                continue

    def start(self):
        """Starts the audio stream and processing thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._process_audio)
        
        self.stream = sd.InputStream(
            callback=self._audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype='float32'
        )
        self.stream.start()
        self.thread.start()

    def stop(self):
        """Stops the audio stream and processing thread."""
        self.is_running = False
        self.stream.stop()
        self.stream.close()
        self.thread.join()
        print("Streaming stopped.")


# --- Example Usage ---
if __name__ == "__main__":
    try:
        # 1. Initialize your original ASR model
        asr_system = StandaloneASR(model_dir=os.getcwd())

        # 2. Initialize and start the streaming wrapper
        streaming_asr = StreamingASR(asr_system)
        streaming_asr.start()
        print("Press 'q' to stop streaming...")
        # Keep the main thread alive
        while True:
            key = input()
            if key.lower() == 'q':
                streaming_asr.stop()
                sys.exit(1)
    
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'streaming_asr' in locals() and streaming_asr.is_running:
            streaming_asr.stop()
            sys.exit(1)