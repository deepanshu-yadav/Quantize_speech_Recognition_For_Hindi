from inference_ctc_float16_non_streaming import StandaloneASRCTC
import os

asr = StandaloneASRCTC(model_dir=os.getcwd())
transcripts = asr.transcribe_file("file.wav")
print(transcripts)