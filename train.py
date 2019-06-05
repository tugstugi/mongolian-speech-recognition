#!/usr/bin/env python

"""Train the Speech2Text network."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import sys
import time
import argparse
from tqdm import *

from apex import amp
import albumentations as A

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from tensorboardX import SummaryWriter

# project imports
from datasets import *
from models import *
from utils import get_last_checkpoint_file_name, load_checkpoint, save_checkpoint

from decoder import GreedyDecoder

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", choices=['librispeech', 'mbspeech', 'bolorspeech'], default='bolorspeech', help='dataset name')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--batch-size", type=int, default=44, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=8, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=1e-5, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--model", choices=['jasper', 'w2l', 'crnn'], default='w2l',
                    help='choices of neural network')
parser.add_argument("--lr", type=float, default=5e-3, help='learning rate for optimization')
parser.add_argument('--mixed-precision', action='store_true', help='enable mixed precision training')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if not use_gpu:
    print("GPU not available!")
    sys.exit(1)
torch.backends.cudnn.benchmark = True

train_transform = Compose([LoadMagSpectrogram(),
                           AddNoiseToMagSpectrogram(noise=NoiseDataset(), probability=0.5),
                           ShiftSpectrogramAlongFrequencyAxis(frequency_shift_max_percentage=0.1, probability=0.7),
                           ComputeMelSpectrogramFromMagSpectrogram(),
                           ApplyAlbumentations(A.Compose([
                               # A.OneOf([A.Blur(blur_limit=3), A.MedianBlur(blur_limit=3)]),  # sometimes hurts, sometimes OK
                               A.Cutout(num_holes=10)  # dataset dependent, longer audios more cutout
                           ], p=1)),
                           TimeScaleSpectrogram(max_scale=0.1, probability=0.5),  # only tiny effect
                           MaskSpectrogram(frequency_mask_max_percentage=0.3,
                                           time_mask_max_percentage=0.1,
                                           probability=1),
                           ShiftSpectrogramAlongTimeAxis(time_shift_max_percentage=0.05, probability=0.7),
                           ])
valid_transform = Compose([LoadMagSpectrogram(), ComputeMelSpectrogramFromMagSpectrogram()])

if args.dataset == 'librispeech':
    from datasets.libri_speech import LibriSpeech as SpeechDataset, vocab

    max_duration = 16.7
    train_dataset = ConcatDataset([
        SpeechDataset(name='train-clean-100', max_duration=max_duration, transform=train_transform),
        SpeechDataset(name='train-clean-360', max_duration=max_duration, transform=train_transform),
        SpeechDataset(name='train-other-500', max_duration=max_duration, transform=train_transform)
    ])
    valid_dataset = SpeechDataset(name='dev-clean', transform=valid_transform)
elif args.dataset == 'bolorspeech':
    from datasets.bolor_speech import BolorSpeech as SpeechDataset, vocab

    max_duration = 16.7
    train_dataset = ConcatDataset([
        SpeechDataset(name='train', max_duration=max_duration, transform=train_transform),
        NoiseDataset(size=5000, transform=train_transform)])
    valid_dataset = SpeechDataset(name='test', transform=valid_transform)
else:
    from datasets.mb_speech import MBSpeech as SpeechDataset, vocab

    train_dataset = SpeechDataset(transform=train_transform)
    valid_dataset = SpeechDataset(transform=valid_transform)
    indices = list(range(len(train_dataset)))
    train_dataset = Subset(train_dataset, indices[:-args.batch_size])
    valid_dataset = Subset(valid_dataset, indices[-args.batch_size:])

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.dataload_workers_nums)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.dataload_workers_nums)

# NN model
if args.model == 'jasper':
    model = TinyJasper(vocab)
elif args.model == 'w2l':
    model = TinyWav2Letter(vocab)
else:
    model = Speech2TextCRNN(vocab)
model = model.cuda()

# loss function
# pytorch master already implemented the CTC loss but not usable yet!
# so we are using now warpctc_pytorch
from warpctc_pytorch import CTCLoss

criterion = CTCLoss()
decoder = GreedyDecoder(labels=vocab)

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.mixed_precision:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

start_timestamp = int(time.time() * 1000)
start_epoch = 0
global_step = 0

logname = "%s_%s_wd%.0e" % (args.dataset, args.model, args.weight_decay)
if args.comment:
    logname = "%s_%s" % (logname, args.comment.replace(' ', '_'))
logdir = os.path.join('logdir', logname)
writer = SummaryWriter(log_dir=logdir)

# load the last checkpoint if exists
last_checkpoint_file_name = get_last_checkpoint_file_name(logdir)
if last_checkpoint_file_name:
    print("loading the last checkpoint: %s" % last_checkpoint_file_name)
    start_epoch, global_step = load_checkpoint(last_checkpoint_file_name, model, optimizer, use_gpu)


def get_lr():
    return optimizer.param_groups[0]['lr']


def lr_decay(step, warmup_steps=2000):
    # https://github.com/tensorflow/tensor2tensor/issues/280
    new_lr = args.lr * warmup_steps ** 0.5 * min((step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
    optimizer.param_groups[0]['lr'] = new_lr


def train(epoch, phase='train'):
    global global_step

    lr_decay(global_step)
    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))

    model.train() if phase == 'train' else model.eval()
    torch.set_grad_enabled(True) if phase == 'train' else torch.set_grad_enabled(False)
    data_loader = train_data_loader if phase == 'train' else valid_data_loader

    it = 0
    running_loss = 0.0
    total_cer, total_wer = 0, 0

    pbar = tqdm(data_loader, unit="audios", unit_scale=data_loader.batch_size)
    for batch in pbar:
        inputs, targets = batch['input'], batch['target']
        inputs_length, targets_length = batch['input_length'], batch['target_length']
        # inputs = inputs.permute(0, 2, 1)

        B, n_feature, T = inputs.size()  # number of feature bins and time
        _, N = targets.size()  # batch size and text count

        # warpctc_pytorch wants Int instead of Long!
        targets = targets.int()
        inputs_length = inputs_length.int()
        targets_length = targets_length.int()

        inputs = inputs.cuda()

        # BxCxT
        outputs = model(inputs)
        if args.model in ['jasper', 'w2l']:
            # make TxBxC
            outputs = outputs.permute(2, 0, 1)
        else:
            # TODO: avoids NaN on CRNN
            inputs_length = torch.IntTensor([outputs.size(0)] * B)

        # warpctc_pytorch wants one dimensional vector without blank elements
        targets_1d = targets.view(-1)
        targets_1d = targets_1d[targets_1d.nonzero().squeeze()]

        # warpctc_pytorch wants the last 3 parameters on the CPU! -> only inputs is converted to CUDA
        loss = criterion(outputs, targets_1d, inputs_length, targets_length)
        loss = loss / B

        if phase == 'train':
            lr_decay(global_step)
            optimizer.zero_grad()

            if args.mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # optimizer.clip_master_grads(100)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

            optimizer.step()

        global_step += 1
        it += 1

        loss = loss.item()
        running_loss += loss

        if global_step % 10 == 0:
            writer.add_scalar('%s/loss' % phase, loss, global_step)
            if phase == 'train':
                writer.add_scalar('%s/learning_rate' % phase, get_lr(), global_step)

        if phase == 'train' and global_step % 50 == 1 or phase == 'valid':
            with torch.no_grad():
                target_strings = decoder.convert_to_strings(targets)
                decoded_output, _ = decoder.decode(outputs.softmax(2).permute(1, 0, 2))
                writer.add_text('%s/prediction' % phase,
                                'truth: %s\npredicted: %s' % (target_strings[0][0], decoded_output[0][0]), global_step)

                if phase == 'valid':
                    cer, wer = 0, 0
                    for x in range(len(target_strings)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        cer += decoder.cer(transcript, reference) / float(len(reference))
                        wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                    total_cer += cer
                    total_wer += wer

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it)
        })

    epoch_loss = running_loss / it
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    if phase == 'valid':
        valid_dataset_length = len(valid_dataset)
        writer.add_scalar('%s/epoch_cer' % phase, (total_cer / valid_dataset_length) * 100, epoch)
        writer.add_scalar('%s/epoch_wer' % phase, (total_wer / valid_dataset_length) * 100, epoch)

        save_checkpoint(logdir, epoch, global_step, model, optimizer)

    return epoch_loss


since = time.time()
epoch = start_epoch
while True:
    train_epoch_loss = train(epoch, phase='train')
    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60,
                                                                     time_elapsed % 60)
    print("train epoch loss %f, step=%d, %s" % (train_epoch_loss, global_step, time_str))

    valid_epoch_loss = train(epoch, phase='valid')
    print("valid epoch loss %f" % valid_epoch_loss)

    epoch += 1
    # if global_step >= hp.model_max_iteration:
    #    print("max step %d (current step %d) reached, exiting..." % (hp.model_max_iteration, global_step))
    #    sys.exit(0)
