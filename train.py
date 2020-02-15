#!/usr/bin/env python

"""Train the Speech2Text network."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import time
import argparse
from tqdm import *

from apex.parallel import DistributedDataParallel
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
parser.add_argument("--dataset", choices=['librispeech', 'mbspeech', 'bolorspeech'], default='bolorspeech',
                    help='dataset name')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--train-batch-size", type=int, default=44, help='train batch size')
parser.add_argument("--valid-batch-size", type=int, default=22, help='valid batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=1e-5, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--model", choices=['jasper', 'w2l', 'crnn'], default='crnn',
                    help='choices of neural network')
parser.add_argument("--lr", type=float, default=7e-3, help='learning rate for optimization')
parser.add_argument('--mixed-precision', action='store_true', help='enable mixed precision training')
parser.add_argument('--warpctc', action='store_true', help='use SeanNaren/warp-ctc instead of torch.nn.CTCLoss')
parser.add_argument('--cudnn-benchmark', action='store_true', help='enable CUDNN benchmark')
parser.add_argument("--max-epochs", default=300, type=int)
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
torch.backends.cudnn.benchmark = args.cudnn_benchmark

train_transform = Compose([LoadMagSpectrogram(),
                           AddNoiseToMagSpectrogram(noise=ColoredNoiseDataset(), probability=0.5),
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
        SpeechDataset(name='annotation', max_duration=max_duration, transform=train_transform),
        SpeechDataset(name='demo', max_duration=max_duration, transform=train_transform),
        ColoredNoiseDataset(size=5000, transform=train_transform),
        BackgroundSounds(size=1000, transform=train_transform)
    ])
    valid_dataset = SpeechDataset(name='test', transform=valid_transform)
else:
    from datasets.mb_speech import MBSpeech as SpeechDataset, vocab

    train_dataset = SpeechDataset(transform=train_transform)
    valid_dataset = SpeechDataset(transform=valid_transform)
    indices = list(range(len(train_dataset)))
    train_dataset = Subset(train_dataset, indices[:-args.valid_batch_size])
    valid_dataset = Subset(valid_dataset, indices[-args.valid_batch_size:])

train_data_sampler, valid_data_sampler = None, None
if args.distributed:
    train_data_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_data_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(train_data_sampler is None),
                               collate_fn=collate_fn, num_workers=args.dataload_workers_nums,
                               sampler=train_data_sampler)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=args.dataload_workers_nums,
                               sampler=None)

if args.model == 'jasper':
    model = TinyJasper(vocab)
elif args.model == 'w2l':
    model = TinyWav2Letter(vocab)
else:
    model = Speech2TextCRNN(vocab)
model = model.cuda()

if args.warpctc:
    from warpctc_pytorch import CTCLoss

    criterion = CTCLoss(blank=0, size_average=False, length_average=False)
else:
    from torch.nn import CTCLoss

    criterion = CTCLoss(blank=0, reduction='sum', zero_infinity=True)

decoder = GreedyDecoder(labels=vocab)

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.mixed_precision:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
if args.distributed:
    model = DistributedDataParallel(model)

start_timestamp = int(time.time() * 1000)
start_epoch = 0
global_step = 0

logname = "%s_%s_wd%.0e" % (args.dataset, args.model, args.weight_decay)
if args.comment:
    logname = "%s_%s" % (logname, args.comment.replace(' ', '_'))
logdir = os.path.join('logdir', logname)
writer = SummaryWriter(log_dir=logdir)
if args.local_rank == 0:
    print(vars(args))
    writer.add_hparams(vars(args), {})

# load the last checkpoint if exists
last_checkpoint_file_name = get_last_checkpoint_file_name(logdir)
if last_checkpoint_file_name:
    print("loading the last checkpoint: %s" % last_checkpoint_file_name)
    start_epoch, global_step = load_checkpoint(last_checkpoint_file_name, model, optimizer, use_gpu=True)


def get_lr():
    return optimizer.param_groups[0]['lr']


def lr_decay(step, warmup_steps=2000):
    # https://github.com/tensorflow/tensor2tensor/issues/280
    new_lr = args.lr * warmup_steps ** 0.5 * min((step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
    optimizer.param_groups[0]['lr'] = new_lr


def train(epoch, phase='train'):
    global global_step

    lr_decay(global_step)
    if args.local_rank == 0:
        print("epoch %3d with lr=%.02e" % (epoch, get_lr()))

    if args.distributed:
        train_data_sampler.set_epoch(epoch)

    model.train() if phase == 'train' else model.eval()
    torch.set_grad_enabled(True) if phase == 'train' else torch.set_grad_enabled(False)
    data_loader = train_data_loader if phase == 'train' else valid_data_loader

    it = 0
    running_loss = 0.0
    total_cer, total_wer = 0, 0

    pbar = None
    if args.local_rank == 0:
        batch_size = args.train_batch_size if phase == 'train' else args.valid_batch_size
        pbar = tqdm(data_loader, unit="audios", unit_scale=batch_size)

    for batch in data_loader if pbar is None else pbar:
        inputs, targets = batch['input'], batch['target']
        inputs_length, targets_length = batch['input_length'], batch['target_length']
        # inputs = inputs.permute(0, 2, 1)

        # warpctc wants Int instead of Long
        targets = targets.int() if args.warpctc else targets.long()
        inputs_length = inputs_length.int() if args.warpctc else inputs_length.long()
        targets_length = targets_length.int() if args.warpctc else targets_length.long()

        B, n_feature, T = inputs.size()  # number of feature bins and time
        _, N = targets.size()  # batch size and text count

        # mix inputs
        # index = np.random.permutation(B)
        # inputs = inputs + random.uniform(0.05, 0.2) * inputs[index]

        # BxCxT
        outputs = model(inputs.cuda())
        if args.model in ['jasper', 'w2l']:
            # make TxBxC
            outputs = outputs.permute(2, 0, 1)
        else:
            # TODO: avoids NaN on CRNN
            inputs_length[:] = outputs.size(0)

        if args.warpctc:
            # warpctc wants one dimensional vector without blank elements
            targets_1d = targets.view(-1)
            targets_1d = targets_1d[targets_1d.nonzero().squeeze()]
            # warpctc wants targets, inputs_length, targets_length on CPU -> don't need to convert to CUDA
            loss = criterion(outputs, targets_1d, inputs_length, targets_length)
        else:
            # nn.CTCLoss wants log softmax
            loss = criterion(outputs.log_softmax(dim=2), targets.cuda(), inputs_length.cuda(), targets_length.cuda())
        loss = loss / B

        if phase == 'train':
            lr_decay(global_step)
            optimizer.zero_grad()

            if args.mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

            optimizer.step()

        global_step += 1
        it += 1

        loss = loss.item()
        running_loss += loss

        if args.local_rank == 0:
            if global_step % 10 == 0:
                writer.add_scalar('%s/loss' % phase, loss, global_step)
                if phase == 'train':
                    writer.add_scalar('%s/learning_rate' % phase, get_lr(), global_step)

            if phase == 'train' and global_step % 50 == 1 or phase == 'valid':
                with torch.no_grad():
                    target_strings = decoder.convert_to_strings(targets)
                    decoded_output, _ = decoder.decode(outputs.softmax(dim=2).permute(1, 0, 2))
                    writer.add_text('%s/prediction' % phase,
                                    'truth: %s\npredicted: %s' % (target_strings[0][0], decoded_output[0][0]),
                                    global_step)

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

    if args.local_rank == 0:
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

        if phase == 'valid':
            valid_dataset_length = len(valid_dataset)
            writer.add_scalar('%s/epoch_cer' % phase, (total_cer / valid_dataset_length) * 100, epoch)
            writer.add_scalar('%s/epoch_wer' % phase, (total_wer / valid_dataset_length) * 100, epoch)
            print('%s/epoch_wer' % phase, (total_wer / valid_dataset_length) * 100)

            save_checkpoint(logdir, epoch, global_step, model, optimizer)

    return epoch_loss


since = time.time()
epoch = start_epoch
while True:
    train_epoch_loss = train(epoch, phase='train')
    if args.local_rank == 0:
        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600,
                                                                         time_elapsed % 3600 // 60,
                                                                         time_elapsed % 60)
        print("train epoch loss %f, step=%d, %s" % (train_epoch_loss, global_step, time_str))
        valid_epoch_loss = train(epoch, phase='valid')
        print("valid epoch loss %f" % valid_epoch_loss)

    epoch += 1

    if epoch > args.max_epochs:
        break
