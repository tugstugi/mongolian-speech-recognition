#!/usr/bin/env python

"""Eval the speech model."""
__author__ = 'Erdene-Ochir Tuguldur'

import argparse
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import *
from models import *
from models.crnn import *
from utils import load_checkpoint

from decoder import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=['librispeech', 'mbspeech', 'bolorspeech', 'kazakh20h'],
                        default='bolorspeech', help='dataset name')
    parser.add_argument("--checkpoint", type=str, required=True, help='checkpoint file to test')
    parser.add_argument("--model", choices=['crnn', 'quartznet5x5', 'quartznet10x5', 'quartznet15x5'], default='crnn',
                        help='choices of neural network')
    parser.add_argument("--normalize", choices=['all_features', 'per_feature'], default='all_features',
                        help="feature normalization")
    parser.add_argument("--batch-size", type=int, default=1, help='batch size')
    parser.add_argument("--dataload-workers-nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument("--lm", type=str, required=False, help='link to KenLM 5-gram binary language model')
    parser.add_argument("--alpha", type=float, default=0.3, help='alpha for CTC decode')
    parser.add_argument("--beta", type=float, default=1.85, help='beta for CTC decode')
    parser.add_argument("--output", type=str, help='eval file container WER/CER for each transcription')
    args = parser.parse_args()
    print(args)

    num_features = 64
    eps = 2 ** -24
    if args.model == 'crnn':
        # CRNN supports only 32 features
        num_features = 32
        eps = 1e-20

    valid_transform = Compose([LoadMagSpectrogram(),
                               ComputeMelSpectrogramFromMagSpectrogram(num_features=num_features,
                                                                       normalize=args.normalize, eps=eps)])
    if args.dataset == 'librispeech':
        from datasets.libri_speech import LibriSpeech as SpeechDataset, vocab

        valid_dataset = SpeechDataset(name='dev-clean', transform=valid_transform)
    elif args.dataset == 'kazakh20h':
        from datasets.kazakh20h_speech import Kazakh20hSpeech as SpeechDataset, vocab

        valid_dataset = SpeechDataset(name='test', transform=valid_transform)
    else:
        from datasets.bolor_speech import BolorSpeech as SpeechDataset, vocab

        valid_dataset = SpeechDataset(name='test', transform=valid_transform)

    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=collate_fn, num_workers=args.dataload_workers_nums)

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

    greedy_decoder = GreedyDecoder(labels=vocab)
    if args.lm:
        t = time.time()
        decoder = BeamCTCDecoder(labels=vocab, num_processes=4,
                                 lm_path=args.lm,
                                 alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=40, cutoff_prob=1.0, beam_width=100)
        print("LM load time: %0.2f" % (time.time() - t))
    else:
        decoder = greedy_decoder

    it = 0
    total_cer, total_wer = 0, 0

    output = None
    if args.output is not None:
        output = open(args.output, 'wt')
    t = time.time()
    pbar = tqdm(valid_data_loader, unit="audios", unit_scale=valid_data_loader.batch_size)
    for batch in pbar:
        inputs, inputs_length, targets = batch['input'], batch['input_length'], batch['target']
        # inputs = inputs.permute(0, 2, 1)
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
            inputs_length = inputs_length.cuda()

        if args.model == 'crnn':
            outputs = model(inputs)
        else:
            outputs, inputs_length = model(inputs, inputs_length)
            # BxCxT -> TxBxC
            outputs = outputs.permute(2, 0, 1)
        it += 1

        target_strings = greedy_decoder.convert_to_strings(targets)
        decoded_output, _ = decoder.decode(outputs.softmax(2).permute(1, 0, 2))

        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            cer = decoder.cer(transcript, reference) / float(len(reference))
            wer = decoder.wer(transcript, reference) / float(len(reference.split()))
            total_cer += cer
            total_wer += wer
            if output is not None:
                output.write("%s,%s,%.2f,%.2f\n" % (valid_dataset.fnames[it + x], reference, cer, wer))

    print('total time: %.2fs' % (time.time() - t))
    print('total CER: %.2f' % (total_cer / len(valid_dataset) * 100))
    print('total WER: %.2f' % (total_wer / len(valid_dataset) * 100))
    if output is not None:
        output.close()
