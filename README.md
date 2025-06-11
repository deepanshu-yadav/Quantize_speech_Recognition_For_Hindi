Kudos to AI4Bharat for training hindi specific speech recognition model.
Visit: https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large


This repository aims to 
1. quantize the .nemo model.
2. remove nemo specific dependencies
3. finally use the converted onnx model. 


---

Note : We have only used the CTC version for this model. 
If you want to use RNNT version then you make trivial changes in the `onnxconversion.ipynb`
notebook.
---

There is a notebook already provided for conversion to float 16 model. 
The name of the notebook is `onnxconversion.ipynb`


# How to perform inference 

Install the depedencies

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

After that install from requirements file

```
pip install -r requirements.txt
```

Now we can run inference 

`python inference.py`

Note a sample file has already been provided. 

Expected Output: 


```
Audio features shape: (1, 80, 1413), Length: [1413]
Transcription: शिवपाल की यह टिप्पणी फ़िल्म काल्या के डायलॉग से मिलतीजुलती है शिवपाल चाहते हैं कि मुलायम पारती के मुखिया फिर से बने फ़िलहाल सपा अध्यक्ष अखिलेश यादव हैं पिता से पार्ट की कमान छीनी थी
```
