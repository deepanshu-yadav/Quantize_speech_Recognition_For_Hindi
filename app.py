import os
import signal
import sys
import threading
import numpy as np
import torch
import sounddevice as sd
from queue import Queue
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import tempfile
import aiofiles
from inference_rnnt_float16_non_streaming import StandaloneASR

# Initialize FastAPI app
app = FastAPI(title="Speech-to-Text API", version="1.0.0")

# Initialize ASR model
try:
    asr = StandaloneASR(model_dir=os.getcwd())
except Exception as e:
    raise RuntimeError(f"Failed to initialize ASR model: {str(e)}")

class StreamingASR:
    def __init__(self, asr_model: StandaloneASR, sample_rate=16000, chunk_size=512, silence_threshold=0.5):
        self.asr_model = asr_model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold_seconds = silence_threshold
        self.audio_queue = Queue()
        self.is_running = False
        self.lock = threading.Lock()

        # Load Silero VAD model
        try:
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True
            )
            self.get_speech_timestamps, _, self.read_audio, _, _ = self.vad_utils
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model: {str(e)}")

    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice audio input."""
        if status:
            print(status, file=sys.stderr)
        with self.lock:
            if self.is_running:
                self.audio_queue.put(indata.copy())

    def _process_audio(self):
        """Process audio chunks for VAD and transcription."""
        audio_buffer = []
        is_speaking = False
        silent_chunks = 0
        num_chunks_for_silence = int((self.sample_rate * self.silence_threshold_seconds) / self.chunk_size)

        while self.is_running:
            try:
                # Get audio chunk non-blocking
                audio_float32 = self.audio_queue.get_nowait()
                audio_tensor = torch.from_numpy(audio_float32).float()

                # Get VAD probability
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

                if speech_prob > 0.5:  # Speech detected
                    if not is_speaking:
                        is_speaking = True
                    audio_buffer.append(audio_float32)
                    silent_chunks = 0
                else:
                    if is_speaking:
                        silent_chunks += 1
                        if silent_chunks > num_chunks_for_silence:
                            is_speaking = False
                            full_utterance = np.concatenate(audio_buffer)
                            audio_buffer = []
                            try:
                                text = self.asr_model.transcribe(full_utterance)
                                yield f"data: {json.dumps({'transcription': text})}\n\n"
                            except Exception as e:
                                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except Queue.Empty:
                continue

    def start(self):
        """Start audio stream and processing."""
        with self.lock:
            if self.is_running:
                return
            self.is_running = True

        self.stream = sd.InputStream(
            callback=self._audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype='float32'
        )
        self.stream.start()

    def stop(self):
        """Stop audio stream and processing."""
        with self.lock:
            if not self.is_running:
                return
            self.is_running = False
        self.stream.stop()
        self.stream.close()
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Queue.Empty:
                break

    def stream_transcriptions(self):
        """Generator for streaming transcriptions as SSE."""
        try:
            self.start()
            yield f"data: {json.dumps({'status': 'Streaming started'})}\n\n"
            for event in self._process_audio():
                yield event
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            self.stop()
            yield f"data: {json.dumps({'status': 'Streaming stopped'})}\n\n"

# Global StreamingASR instance
streaming_asr = None

def signal_handler(sig, frame):
    """Handle SIGINT and SIGTERM."""
    global streaming_asr
    if streaming_asr:
        streaming_asr.stop()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.on_event("startup")
async def startup_event():
    """Initialize StreamingASR on startup."""
    global streaming_asr
    streaming_asr = StreamingASR(asr_model=asr)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global streaming_asr
    if streaming_asr:
        streaming_asr.stop()

@app.post("/transcribe")
async def transcribe_file_endpoint(file: UploadFile = File(...)):
    """Endpoint to transcribe an uploaded audio file."""
    if not file.filename.endswith(('.wav', '.flac', '.mp3')):
        raise HTTPException(status_code=400, detail="Unsupported audio format. Use WAV, FLAC, or MP3.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            async with aiofiles.open(temp_file.name, 'wb') as f:
                content = await file.read()
                await f.write(content)
            transcription = asr.transcribe_file(temp_file.name)
        os.unlink(temp_file.name)
        return JSONResponse({"transcription": transcription})
    except Exception as e:
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/stream/transcribe")
async def stream_transcribe():
    """Endpoint for streaming transcriptions via Server-Sent Events."""
    return StreamingResponse(
        streaming_asr.stream_transcriptions(),
        media_type="text/event-stream"
    )