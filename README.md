Mongolian speech recognition experiments using 5 hours audio from first 3 books of the
[Mongolian Bible](https://www.bible.com/mn/versions/1590-2013-ariun-bibli-2013).
This dataset was already successfully used to create a [Mongolian text-to-speech system](https://github.com/tugstugi/pytorch-dc-tts).

Because of the dataset size, only cut down versions of the following papers are implemented:
* [Letter-Based Speech Recognition with Gated ConvNets](https://arxiv.org/abs/1712.09444)
* ...

## Training
1. Install the `warp-ctc`python binding: https://github.com/SeanNaren/warp-ctc
2. Install remaining dependencies: `pip install -r requirements.txt`
3. Download the dataset: `python dl_mbspeech.py`
4. Train: `python train.py`
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

The dataset contains only first 3 books of the Mongolian Bible. You can validate your trained model
from other Bible books (download them from https://www.bible.com/versions/1590-2013-ariun-bibli-2013 as mp3 file).

To validate an audio file using a pretrained model, use following commands:
```
# download a pretrained model
wget https://www.dropbox.com/s/9wan945h110wmyc/epoch-0182-fb4c392.pth
# switch to the commit where the model was trained
git checkout fb4c392
# evaluate an audio file
python eval.py --checkpoint=epoch-0182-fb4c392.pth test.mp3
```

For fun, you can also generate an audio with a Mongolian TTS and try to recognize it.
The following code generates an audio with the
[TTS of the Mongolian National University](http://172.104.34.197/nlp-web-demo/)
and does speech recognition on that generated audio:
```
# generate audio for 'Миний төрсөн нутаг Монголын сайхан орон'
wget -O test.wav "http://172.104.34.197/nlp-web-demo/tts?voice=1&text=Миний төрсөн нутаг Монголын сайхан орон."
# speech recognition on that TTS generated audio
python eval.py --checkpoint=epoch-0182-fb4c392.pth test.wav
# will output: 'биний төрсн нуутөр мөнголын сайхон орн'
```

## TODO
Одоогийн байдлаар энд бичсэн кодыг ашиглан нэг хүний дуу хоолойг ямар ч асуудалгүйгээр таньж
текст рүү хөрвүүлэх боломжтой. Үүнийг сайжруулж дурын хүний дуу хоолойг таниулдаг болгохын тулд
доорх ажлуудыг хийх ёстой:
1. Олон хүний дуу хоолой агуулсан ярианы сан үүсгэх. Яаг үүнд зориулаад Mozilla
[Common Voice](https://voice.mozilla.org/en/languages) гэж төсөл эхлүүлсэн байгаа бөгөөд Англи
хэлнээс гадна Хятад, Турк, Киргиз гэх мэт хэл орсон. Энд Монгол хэлийг оруулбал дуртай хүмүүс нь
Common Voice-ын цахим хуудас болон гар утасны аппуудаар нь ороод өөрийн дуу хоолойгоо бичээд ярианы сан үүсгэж болно.
2. [Language model](https://en.wikipedia.org/wiki/Language_model) үүсгэх. Үүнийг хар ухаанаар
тайлбарлавал баруун аймгийн хүн `көк тэнгэр` гэж хэлбэл `хөх тэнгэр` гэж хэлэх гэсэн байна аа гэж таниад засдаг
model юм. Үүнийг хийхэд том хэмжээний текст корпус шаардлагатай. Эхний ээлжид Википедиагийн
Монгол агуулгуудыг хэрэглээд [KenLM](https://github.com/kpu/kenlm) ашиглаж language model үүсгэнэ.

Дээрх ажлуудыг хийхэд туслах мөн code contribute хийх хүн байвал tugstugi AT gmail.com хаягаар холбогдоно уу.
