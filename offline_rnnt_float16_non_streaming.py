from inference_rnnt_float16_non_streaming import StandaloneASRRNNT
import os

asr = StandaloneASRRNNT(model_dir=os.getcwd())
transcripts = asr.transcribe_file("file.wav")
print(transcripts)