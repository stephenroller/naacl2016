#!/usr/bin/env python

import sys
import argparse
import os.path
from datetime import datetime, timedelta

import numpy as np
from theano.printing import debugprint
from sklearn.preprocessing import normalize

from debug import debug_feedforward_pprint
from utdeftvs import VectorSpace, load_numpy
from ioutil import CorpusBatchReader, DataIterator, CSVLogger, logger
from eval import intrinsic_eval
import models

# CONSTANTS

LEARNING_RATE = 0.01
SPACE_FILENAME = "lexsub_embeddings.npz"
CORPUS_FOLDER = "lexsub_embeddings.ukwac.sparse.rels512.min2"

EPOCHS = 10
BATCH_SIZE = 256

DEBUG = False

def _dictmerge(a, b):
    c = {}
    for k, v in a.iteritems():
        c[k] = v
    for k, v in b.iteritems():
        c[k] = v
    return c

def _compute_eta(start, progress):
    delta = datetime.now() - start
    full_seconds = delta.total_seconds() * (1. / (progress + 1e-6))
    eta_seconds = np.floor((1 - progress) * full_seconds)
    return timedelta(seconds=eta_seconds)

def _generate_filename(modelinfo):
    dt = datetime.now().strftime("%Y%m%d")
    fmt = "mn=%(model)s__d=%(dimensions)04d__h=%(hidden)04d__lr=%(learningrate)f" % modelinfo
    return dt + "__" + fmt


def main():
    parser = argparse.ArgumentParser('description')
    parser.add_argument('--logfolder', '-l', help='Log folder.')
    parser.add_argument('--csvfolder', '-c', help='Output CSV folder for graphs.')
    parser.add_argument('--output', '-o', help='Folder for saving output models.')
    parser.add_argument('--model', '-m', help='Selects a particular model.')
    parser.add_argument('--maxbatches', '-B', default=0, type=int, help='Maximum number of batches to process (in thousands).')
    parser.add_argument('--batchsize', '-b', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--dimensions', '-d', type=int, default=0, help='Number of dimensions from the space to use. If 0 (default), use all.')
    parser.add_argument('--learningrate', '-r', type=float, default=LEARNING_RATE, help='Learning rate')
    args = parser.parse_args()

    logger.debug("Reading distributional space '%s'" % SPACE_FILENAME)
    space = load_numpy(SPACE_FILENAME, insertblank=True)
    if args.dimensions:
        space.matrix = space.matrix[:,:args.dimensions]
    if True:
        m = space.matrix
        norm_mean = m[1:].mean(axis=0)
        norm_std = (m[1:].std(axis=0) * 10)
        m = (m - norm_mean) / norm_std
        m[0] = 0
        space.matrix = m
    #space = space.normalize()
    logger.debug("Finished reading space")
    logger.debug("Space contains %d words with %d dimensions each." % space.matrix.shape)

    cbr = CorpusBatchReader(CORPUS_FOLDER, space, batch_size=args.batchsize)
    data_iterator = DataIterator(cbr, epochs=1, maxbatches=args.maxbatches * 1000)

    HIDDEN = space.matrix.shape[1]

    logger.debug("Compiling compute graph")
    R = data_iterator.test[0].shape[1]
    model = models.get_model(args.model, space, R, HIDDEN, args.learningrate)

    modelinfo = {
        'model': args.model,
        'learningrate': args.learningrate,
        'hidden': HIDDEN,
        'space': SPACE_FILENAME,
        'dimensions': space.matrix.shape[1],
    }

    filename = _generate_filename(modelinfo)
    csvlog = CSVLogger(os.path.join(args.csvfolder, filename + ".csv"))

    logger.debug("Compilation finished")
    if DEBUG:
        logger.debug("Theano compute graph:\n" + debugprint(model._train.maker.fgraph.outputs[0], file='str'))

    logger.debug("Starting training")
    start_time = datetime.now()
    for X, Y in data_iterator:
        trainscore = model.train_on_batch(X, Y)

        if data_iterator.batch % 1000 == 0:
            valscore = model.evaluate(*data_iterator.val, verbose=False)
            testscore = model.evaluate(*data_iterator.test, verbose=False)
            progress = data_iterator.progress()
            elapsed = (datetime.now() - start_time)
            rank = intrinsic_eval(model, space, data_iterator.test[0], data_iterator.test[1])
            #rank = 0.0
            eta = _compute_eta(start_time, progress)
            batchinfo = dict(
                epoch=data_iterator.epoch,
                kbatch=data_iterator.batch/1000,
                trainscore=trainscore,
                valscore=valscore,
                testscore=testscore,
                intrinsic=rank,
                progress=100 * progress,
                elapsed=elapsed.total_seconds(),
                eta=eta
            )
            info = _dictmerge(batchinfo, modelinfo)
            logger.debug("%(epoch)3d ep %(kbatch)8d Kba %(intrinsic)6.4f / %(valscore)8.5f / %(testscore)8.5f [%(progress)5.1f%% eta %(eta)s]" % info)
            del info['eta']
            csvlog.append(info)

        if data_iterator.batch % 5000 == 0:
            checkpoint_filename = os.path.join(args.output, "%s__batch%08d.hd5" % (filename, data_iterator.batch))
            logger.debug("Checkpointing model to %s" % checkpoint_filename)
            model.save_weights(checkpoint_filename, overwrite=True)


if __name__ == '__main__':
    main()

