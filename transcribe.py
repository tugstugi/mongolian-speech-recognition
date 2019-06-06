#!/usr/bin/env python

"""Read a WAV file and try to recognize the speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import argparse
import torch
import time

from datasets import *
from datasets.mb_speech import vocab
from models import *
from models.crnn import *
from utils import load_checkpoint

from decoder import *


def transcribe(data, args):
    use_gpu = torch.cuda.is_available()
    print('use_gpu:', use_gpu)

    if args.model == 'jasper':
        model = TinyJasper(vocab)
    elif args.model == 'w2l':
        model = TinyWav2Letter(vocab)
    else:
        model = Speech2TextCRNN(vocab)
    load_checkpoint(args.checkpoint, model, optimizer=None, use_gpu=use_gpu)
    model.eval()
    model.cuda() if use_gpu else model.cpu()

    inputs = torch.from_numpy(data['input']).unsqueeze(0)
    # inputs = inputs.permute(0, 2, 1)
    if use_gpu:
        inputs = inputs.cuda()

    torch.set_grad_enabled(False)
    t = time.time()
    outputs = model(inputs)
    if args.model in ['jasper', 'w2l']:
        outputs = outputs.permute(2, 0, 1)
    outputs = outputs.softmax(2).permute(1, 0, 2)
    print("inference time: %.3fs" % (time.time() - t))

    greedy_decoder = GreedyDecoder(labels=vocab)
    t = time.time()
    decoded_output, _ = greedy_decoder.decode(outputs)
    print("decode time without LM: %.3fs" % (time.time() - t))
    print("Predicted without LM:")
    print(decoded_output[0][0])

    if args.lm:
        beam_ctc_decoder = BeamCTCDecoder(labels=vocab, num_processes=4,
                                          lm_path='mn_5gram.binary',
                                          alpha=args.alpha, beta=args.beta,
                                          cutoff_top_n=40, cutoff_prob=1.0, beam_width=1000)
        t = time.time()
        decoded_output, _ = beam_ctc_decoder.decode(outputs)
        print()
        print("decode time with LM: %.3fs" % (time.time() - t))
        print("Predicted with LM:")
        print(decoded_output[0][0])

    return decoded_output[0][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help='checkpoint file to test')
    parser.add_argument("--model", choices=['jasper', 'w2l', 'crnn'], default='w2l',
                        help='choices of neural network')
    parser.add_argument("--lm", type=str, required=False, help='link to KenLM 5-gram binary language model')
    parser.add_argument("--alpha", type=float, default=0.3, help='alpha for CTC decode')
    parser.add_argument("--beta", type=float, default=1.85, help='beta for CTC decode')
    parser.add_argument("audio", help='a WAV file')
    args = parser.parse_args()

    data = {
        'fname': args.audio,
        'text': ''
    }
    data = Compose([LoadAudio(), ComputeMagSpectrogram(), ComputeMelSpectrogramFromMagSpectrogram()])(data)

    transcribe(data, args)
