#!/usr/bin/env python

"""Train the Speech2Text network."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import time
import argparse
from tqdm import *

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from tensorboardX import SummaryWriter

# project imports
from datasets import collate_fn, Compose, LoadAudio, SpeedChange, ExtractSpeechFeatures
from models import TinyWav2Letter
from utils import get_last_checkpoint_file_name, load_checkpoint, save_checkpoint

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", choices=['librispeech', 'mbspeech'], default='mbspeech', help='dataset name')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--batch-size", type=int, default=8, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=8, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=0.0000, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--lr", type=float, default=0.0003, help='learning rate for optimization')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True

if args.dataset == 'librispeech':
    from datasets.libri_speech import LibriSpeech as SpeechDataset, vocab, idx2char
else:
    from datasets.mb_speech import MBSpeech as SpeechDataset, vocab, idx2char

train_dataset = SpeechDataset(transform=Compose([LoadAudio(), SpeedChange(), ExtractSpeechFeatures()]))
valid_dataset = SpeechDataset(transform=Compose([LoadAudio(), ExtractSpeechFeatures()]))
indices = list(range(len(train_dataset)))
train_sampler = SubsetRandomSampler(indices[:-args.batch_size])
valid_sampler = SubsetRandomSampler(indices[-args.batch_size:])

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=args.dataload_workers_nums,
                               sampler=train_sampler)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=args.dataload_workers_nums,
                               sampler=valid_sampler)

# NN model
model = TinyWav2Letter(vocab)
if use_gpu:
    model = model.cuda()

# loss function
# pytorch master already implemented the CTC loss but not usable yet!
# so we are using now warpctc_pytorch
from warpctc_pytorch import CTCLoss

criterion = CTCLoss()

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

start_timestamp = int(time.time() * 1000)
start_epoch = 0
global_step = 0

logdir = os.path.join('logdir', args.dataset)
writer = SummaryWriter(log_dir=logdir)

# load the last checkpoint if exists
last_checkpoint_file_name = get_last_checkpoint_file_name(logdir)
if last_checkpoint_file_name:
    print("loading the last checkpoint: %s" % last_checkpoint_file_name)
    start_epoch, global_step = load_checkpoint(last_checkpoint_file_name, model, optimizer)


def get_lr():
    return optimizer.param_groups[0]['lr']


def lr_decay(step, warmup_steps=4000):
    # https://github.com/tensorflow/tensor2tensor/issues/280
    new_lr = args.lr * warmup_steps ** 0.5 * min((step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
    optimizer.param_groups[0]['lr'] = new_lr


def train(train_epoch, phase='train'):
    global global_step

    lr_decay(global_step)
    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))

    model.train() if phase == 'train' else model.eval()
    torch.set_grad_enabled(True) if phase == 'train' else torch.set_grad_enabled(False)
    data_loader = train_data_loader if phase == 'train' else valid_data_loader

    it = 0
    running_loss = 0.0

    pbar = tqdm(data_loader, unit="audios", unit_scale=data_loader.batch_size)
    for batch in pbar:
        inputs, targets = batch['input'], batch['target']
        inputs_length, targets_length = batch['input_length'], batch['target_length']
        inputs = inputs.permute(0, 2, 1)

        B, N = targets.size()  # batch size and text count
        _, n_feature, T = inputs.size()  # number of feature bins and time

        targets.requires_grad = False
        targets_length.requires_grad = False
        inputs_length.requires_grad = False

        # warpctc_pytorch wants Int instead of Long!
        targets = targets.int()
        inputs_length = inputs_length.int()
        targets_length = targets_length.int()

        if use_gpu:
            inputs = inputs.cuda()

        # BxCxT
        outputs = model(inputs)
        # make TxBxC
        outputs = outputs.permute(2, 0, 1)

        # warpctc_pytorch wants one dimensional vector without blank elements
        targets = targets.view(-1)
        targets = targets[targets.nonzero().squeeze()]

        # warpctc_pytorch wants the last 3 parameters on the CPU! -> only inputs is converted to CUDA
        loss = criterion(outputs, targets, inputs_length, targets_length)
        loss = loss / B

        if phase == 'train':
            lr_decay(global_step)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()

        global_step += 1
        it += 1

        loss = loss.item()
        running_loss += loss

        if global_step % 50 == 0:
            writer.add_scalar('%s/loss' % phase, loss, global_step)
            if phase == 'train':
                writer.add_scalar('%s/learning_rate' % phase, get_lr(), global_step)

        if phase == 'train' and global_step % 100 == 1 or phase == 'valid':
            def to_text(tensor, max_length=None, remove_repetitions=False):
                sentence = ''
                sequence = tensor.cpu().detach().numpy()
                for i in range(len(sequence)):
                    if max_length is not None and i >= max_length:
                        continue
                    char = idx2char[sequence[i]]
                    if char != 'B':  # ignore blank
                        if remove_repetitions and i != 0 and char == idx2char[sequence[i - 1]]:
                            pass
                        else:
                            sentence = sentence + char
                return sentence
            prediction = outputs.softmax(2).max(2)[1]
            ground_truth = to_text(targets, targets_length[0])
            predicted_text = to_text(prediction[:, 0], remove_repetitions=True)
            # print(ground_truth)
            # print(predicted_text)
            writer.add_text('%s/prediction' % phase,
                            'truth: %s\npredicted: %s' % (ground_truth, predicted_text), global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it)
        })

    epoch_loss = running_loss / it
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    if phase == 'valid':
        save_checkpoint(logdir, train_epoch, global_step, model, optimizer)

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
