#!/usr/bin/env python
import os
import os.path
import logging
import itertools
import numpy as np

# before anything else, configure the logger
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s   %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

N = 2


class CorpusBatchReader(object):
    def __init__(self, folder, dist_space, batch_size=256):
        self.folder = folder
        self.batch_size = batch_size
        self.dist_space = dist_space

        self.files = os.listdir(folder)
        if not self.files:
            raise ValueError("%s doesn't have any files in it!" % folder)
        self.fileno = 0

        self._read_file()

    def __iter__(self):
        return self

    def _read_file(self):
        logger.debug("Reading file '%s'" % self.files[self.fileno])
        file = self.files[self.fileno]
        # todo use path.join
        compressed = np.load(os.path.join(self.folder, file))
        self._targetids = compressed['targets']
        self._contexts = compressed['contexts'].item(0)
        self._file_idx = 0

    def next(self):
        """
        Returns the next mini-batch.
        """
        # first check if this is the end of the file
        if self._file_idx >= len(self._targetids):
            # go to next file
            self.fileno += 1
            if self.fileno >= len(self.files):
                # out of everything!
                raise StopIteration
            self._read_file()


        batch_size = self.batch_size

        idx = self._file_idx
        self._file_idx += batch_size

        Yids = self._targetids[idx:idx+batch_size]
        #Y = self.dist_space.matrix[Yids]
        X = self._contexts[idx:idx+batch_size,:256].todense()

        return X, np.array([Yids]).T

    def rewind(self):
        self.fileno = 0
        self._read_file()

    def progress(self):
        nof = float(len(self.files))
        floor = self.fileno / nof
        maxchunk = 1./nof
        chunk = self._file_idx / float(len(self._targetids))
        return floor + chunk * maxchunk


class DataIterator(object):
    def __init__(self, corpus_batch_reader, epochs=1, maxbatches=0):
        self.epoch = 0
        self.max_epochs = epochs
        self.cbr = corpus_batch_reader
        self.maxbatches = maxbatches

        self.test = self.cbr.next()
        self.val = self.cbr.next()
        self.train = None

        self.batch = -1

    def __iter__(self):
        return self

    def next(self):
        if self.maxbatches and self.batch >= self.maxbatches:
            raise StopIteration
        self.train = self.val
        try:
            self.val= self.cbr.next()
        except StopIteration:
            self.epoch += 1
            if self.epoch >= self.max_epochs:
                raise
            self.cbr.rewind()
            # don't train on the "test" item; skip one item off!
            self.cbr.next()
            self.val = self.cbr.next()

        self.batch += 1
        return self.train

    def progress(self):
        if self.maxbatches:
            return float(self.batch) / self.maxbatches
        me = float(self.max_epochs)
        floor = self.epoch / me
        maxchunk = 1. / me
        prog = self.cbr.progress()
        chunk = self.cbr.progress() * maxchunk
        return floor + chunk

class CSVLogger(object):
    def __init__(self, filename):
        self.filename = filename
        self.count = 0
        if self.filename:
            self._handle = open(self.filename, 'w')

    def append(self, record):
        if self.filename:
            if not self.count:
                self.keys = sorted(record.keys())
                self._handle.write(",".join(self.keys) + "\n")
            self._handle.write(",".join(str(record.get(k, '')) for k in self.keys))
            self._handle.write("\n")
            self._handle.flush()
        self.count += 1

    def __del__(self):
        if self.filename:
            self._handle.close()

