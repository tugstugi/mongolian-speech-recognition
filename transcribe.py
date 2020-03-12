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


def transcribe(data, num_features, args):
    use_gpu = torch.cuda.is_available()
    print('use_gpu:', use_gpu)

    if args.model == 'quartznet5x5':
        model = QuartzNet5x5(vocab=vocab, num_features=num_features)
    elif args.model == 'quartznet10x5':
        model = QuartzNet10x5(vocab=vocab, num_features=num_features)
    elif args.model == 'quartznet15x5':
        model = QuartzNet15x5(vocab=vocab, num_features=num_features)
    else:
        model = Speech2TextCRNN(vocab)
    load_checkpoint(args.checkpoint, model, optimizer=None, use_gpu=use_gpu, remove_module_keys=True)
    model.eval()
    model.cuda() if use_gpu else model.cpu()
    torch.set_grad_enabled(False)

    inputs = torch.from_numpy(data['input']).unsqueeze(0)
    inputs_length = torch.from_numpy(np.array([data['input_length']])).long()
    # inputs = inputs.permute(0, 2, 1)
    if use_gpu:
        inputs = inputs.cuda()
        inputs_length = inputs_length.cuda()

    t = time.time()
    if args.model == 'crnn':
        outputs = model(inputs)
    else:
        outputs, inputs_length = model(inputs, inputs_length)
        # BxCxT -> TxBxC
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
                                          lm_path=args.lm,
                                          alpha=args.alpha, beta=args.beta,
                                          cutoff_top_n=40, cutoff_prob=1.0, beam_width=100)
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
    parser.add_argument("--model", choices=['crnn', 'quartznet5x5', 'quartznet10x5', 'quartznet15x5'], default='crnn',
                        help='choices of neural network')
    parser.add_argument("--normalize", choices=['all_features', 'per_feature'], default='all_features',
                        help="feature normalization")
    parser.add_argument("--lm", type=str, required=False, help='link to KenLM 5-gram binary language model')
    parser.add_argument("--alpha", type=float, default=0.3, help='alpha for CTC decode')
    parser.add_argument("--beta", type=float, default=1.85, help='beta for CTC decode')
    parser.add_argument("audio", help='a WAV file')
    args = parser.parse_args()
    print(args)

    num_features = 64
    eps = 2 ** -24
    if args.model == 'crnn':
        # CRNN supports only 32 features
        num_features = 32
        eps = 1e-20

    transform = Compose([LoadAudio(), ComputeMagSpectrogram(),
                         ComputeMelSpectrogramFromMagSpectrogram(num_features=num_features,
                                                                 normalize=args.normalize, eps=eps)])
    transcribe(transform({
        'fname': args.audio,
        'text': ''
    }), num_features, args)
