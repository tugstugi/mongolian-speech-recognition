An online demo trained with a Mongolian proprietary dataset (WER 8%): [https://chimege.mn/](https://chimege.mn/).

In this repo, following papers are implemented:
* [QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions](https://arxiv.org/abs/1910.10261)
* [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
  * speech recognition as optical character recognition

This repo is partially based on:
* decoder from [SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
* Jasper/QuartzNet blocks from [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

## Training
1. Install PyTorch>=1.3 with conda
2. Install remaining dependencies: `pip install -r requirements.txt`
3. Download the Mongolian Bible dataset: `cd datasets && python dl_mbspeech.py`
4. Pre compute the mel spectrograms: `python preprop_dataset.py --dataset mbspeech`
5. Train: `python train.py --model crnn --max-epochs 50 --dataset mbspeech --lr-warmup-steps 100`
   * logs for the TensorBoard are saved in the folder `logdir`

## Results
During the training, the ground truth and recognized texts are logged into the TensorBoard.
Because the dataset contains only a single person, the predicted texts from the validation set
should be already recognizable after few epochs:

**EXPECTED:**
```
аливаа цус хувцсан дээр үсрэхэд цус үсэрсэн хэсгийг та нар ариун газарт угаагтун
```
**PREDICTED:**
```
аливаа цус хувцсан дээр үсэрхэд цус усарсан хэсхийг та нар ариун газарт угаагтун
```

For fun, you can also generate an audio with a Mongolian TTS and try to recognize it.
The following code generates an audio with the
[TTS of the Mongolian National University](http://172.104.34.197/nlp-web-demo/)
and does speech recognition on that generated audio:
```
# generate audio for 'Миний төрсөн нутаг Монголын сайхан орон'
wget -O test.wav "http://172.104.34.197/nlp-web-demo/tts?voice=1&text=Миний төрсөн нутаг Монголын сайхан орон."
# speech recognition on that TTS generated audio
python transcribe.py --checkpoint=logdir/mbspeech_crnn_sgd_wd1e-05/epoch-0050.pth --model=crnn test.wav
# will output: 'миний төрсөн нут мөнголын сайхан оөрулн'
```

It is also possible to use a KenLM binary model. First download it from
[tugstugi/mongolian-nlp](https://github.com/tugstugi/mongolian-nlp#mongolian-language-model).
After that, install [parlance/ctcdecode](https://github.com/parlance/ctcdecode). Now you can transcribe with the language model:
```
python transcribe.py --checkpoint=path/to/checkpoint --lm=mn_5gram.binary --alpha=0.3 test.wav
```

## Contribute
If you are Mongolian and want to help us, please record your voice on [Common Voice](https://voice.mozilla.org/mn/speak).
