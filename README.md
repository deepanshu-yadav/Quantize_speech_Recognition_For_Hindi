Kudos to AI4Bharat for training hindi specific speech recognition model.
Visit: https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large


This repository aims to 
1. quantize the .nemo model for both CTC and RNNT versions.
2. remove nemo specific dependencies
3. finally use the converted onnx model for both offline and online(microphone) use.

The hugging face link for this repository is 
https://huggingface.co/pronoobie/indic_conformer_hi_float16_onnx_256_vocab

---
Converted for both CTC  and RNNT versions.

---

There is a notebook already provided for conversion to float 16 model.  
The name of the notebook is `onnxconversionCTC.ipynb` for CTC.
The name of the notebook is `onnxconversionRNNT.ipynb` for RNNT version.


# How to perform inference 

Install the depedencies

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

After that install from requirements file

```
pip install -r requirements.txt
```

## For CTC float16 (non streaming version) offline mode
Now we can run inference 

`python offline_ctc_float16_inference.py`

Note a sample file has already been provided. 

Expected Output: 


```
Audio features shape: (1, 80, 1413), Length: [1413]
Transcription: शिवपाल की यह टिप्पणी फ़िल्म काल्या के डायलॉग से मिलतीजुलती है शिवपाल चाहते हैं कि मुलायम पारती के मुखिया फिर से बने फ़िलहाल सपा अध्यक्ष अखिलेश यादव हैं पिता से पार्ट की कमान छीनी थी
```

## For CTC float16 (non streaming mode) live mode 

You can perform transcription live from your sound device as well.

Execute

`python realtime_ctc_float16_non_streaming.py`


Expected Output

```
Using cache found in C:\Users\DEEPANSHU/.cache\torch\hub\snakers4_silero-vad_master
Listening... (Speak into the microphone)
Press 'q' to stop streaming...
C:\Users\DEEPANSHU\Desktop\automation\speech\hindi\git_inference_push\realtime_ctc_float16_non_streaming.py:55: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:209.)
  audio_tensor = torch.from_numpy(audio_float32)
Speech detected, recording...
Silence detected, transcribing...
Transcription: तो कैसे हैं आप सब
Listening...
Speech detected, recording...
Silence detected, transcribing...
Transcription: आपसे मिल के अच्छा लगा
Listening...
```

## For RNNT  

### For Realtime (microphone) 
It is float 16 rnnt version with non streaming mode. 

`python realtime_rnnt_float16_non_streaming.py`

### Offline file based
It is float 16 rnnt version with non streaming mode. 

`python offline_rnnt_float16_non_streaming.py`