Mongolian speech recognition experiments using 5 hours audio from the [Mongolian Bible](https://www.bible.com/mn/versions/1590-2013-ariun-bibli-2013).
This dataset was already successfully used to create a [Mongolian text-to-speech system](https://github.com/tugstugi/pytorch-dc-tts).

```
Монгол speech recognition-д зориулсан ярианы сан үүсгэхэд хамтрах сонирхолтой
хүн байвал tugstugi AT gmail.com хаягаар холбогдоно уу.
```

Because of the dataset size, only cut down versions of the following papers are implemented:
* [Letter-Based Speech Recognition with Gated ConvNets](https://arxiv.org/abs/1712.09444)
* ...

## Training
1. Install the `warp-ctc`python binding: https://github.com/SeanNaren/warp-ctc
2. Install remaining dependencies: `pip install -r requirements.txt`
3. Download the dataset: `python dl_mbspeech.py`
4. Train: `python train.py`
