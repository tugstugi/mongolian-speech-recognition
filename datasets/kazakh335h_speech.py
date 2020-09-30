"""Kazakh 335h dataset: https://issai.nu.edu.kz/kz-speech-corpus/"""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import csv

from torch.utils.data import Dataset

from .kazakh78h_speech import vocab, idx2char, char2idx, read_metadata


class Kazakh335hSpeech(Dataset):

    def __init__(self, name='train', max_duration=17, transform=None):
        self.transform = transform

        datasets_path = os.path.dirname(os.path.realpath(__file__))
        dataset_path = os.path.join(datasets_path, 'kazakh335h')
        csv_file = os.path.join(dataset_path, 'kazakh335h-%s.csv' % name)
        self.fnames, self.texts = read_metadata(dataset_path, csv_file, max_duration, min_duration=1)

    def __getitem__(self, index):
        data = {
            'fname': self.fnames[index],
            'text': self.texts[index]
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':
    from pydub import AudioSegment
    datasets_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(datasets_path, 'kazakh335h')

    name = 'train'
    reader = csv.reader(open(os.path.join(dataset_path, 'Meta', '%s.csv' % name), 'rt'), delimiter=' ')
    next(reader)  # skip header

    for line in reader:
        fname = os.path.join('Audios', '%s.wav' % line[0])
        text = ''
        with open(os.path.join(dataset_path, 'Transcriptions', '%s.txt' % line[0])) as f:
            text = f.read().strip()

        duration = AudioSegment.from_wav(os.path.join(dataset_path, fname)).duration_seconds
        print("%s,%.2f,%s" % (fname, duration, text))
