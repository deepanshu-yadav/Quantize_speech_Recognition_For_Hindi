from inference_ctc_float16_non_streaming import StandaloneASR
import os

asr = StandaloneASR(model_dir=os.getcwd())
transcripts = asr.transcribe("file.wav")
print(transcripts)