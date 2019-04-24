#!/usr/bin/env python

"""Read a WAV file and try to recognize the speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import argparse
import torch
import time

from datasets import Compose, LoadAudio, ExtractSpeechFeatures
from datasets.mb_speech import vocab
from models import TinyWav2Letter
from utils import load_checkpoint

from decoder import *


def transcribe(data, args):
    use_gpu = torch.cuda.is_available()
    print('use_gpu:', use_gpu)

    model = TinyWav2Letter(vocab)
    load_checkpoint(args.checkpoint, model, optimizer=None, use_gpu=use_gpu)
    model.eval()
    model.cuda() if use_gpu else model.cpu()

    inputs = torch.from_numpy(data['input']).unsqueeze(0)
    inputs = inputs.permute(0, 2, 1)
    if use_gpu:
        inputs = inputs.cuda()

    torch.set_grad_enabled(False)
    t = time.time()
    outputs = model(inputs)
    outputs = outputs.permute(2, 0, 1)
    outputs = outputs.softmax(2).permute(1, 0, 2)
    print("inference time: %.3fs" % (time.time() - t))

    if args.lm:
        decoder = BeamCTCDecoder(labels=vocab, lm_path='mn_5gram.binary', num_processes=4,
                                 alpha=args.alpha, beta=args.beta, cutoff_top_n=40, cutoff_prob=1.0, beam_width=1000)
    else:
        decoder = GreedyDecoder(labels=vocab)

    t = time.time()
    decoded_output, _ = decoder.decode(outputs)
    print("decode time: %.3fs" % (time.time() - t))

    return decoded_output[0][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help='checkpoint file to test')
    parser.add_argument("--lm", type=str, required=False, help='link to KenLM 5-gram binary language model')
    parser.add_argument("--alpha", type=float, default=0.3, help='alpha for CTC decode')
    parser.add_argument("--beta", type=float, default=1.85, help='beta for CTC decode')
    parser.add_argument("audio", help='a WAV file')
    args = parser.parse_args()

    data = {
        'fname': args.audio,
        'text': ''
    }
    data = Compose([LoadAudio(), ExtractSpeechFeatures()])(data)

    result = transcribe(data, args)

    print("Predicted:")
    print(result)
