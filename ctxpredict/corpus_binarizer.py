#!/usr/bin/env python

import sys
import argparse
import bz2

import numpy as np
import scipy
import scipy.sparse

from utdeftvs import load_numpy

def read_relationships(filename, max_relationships):
    labels = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i >= max_relationships:
                break
            labels.append(line.strip().split("\t")[0])
    return {l: i for i, l in enumerate(labels)}

def magic_open(filename):
    if filename == '-':
        return sys.stdin
    elif filename.endswith('.bz2'):
        return bz2.BZ2File(filename)
    else:
        return open(filename)

def pull_items(filename, space, rels):
    with magic_open(filename) as data:
        last = None
        cum = []
        for line in data:
            target, relation, context = line.strip().split("\t")
            try:
                tarid = space.lookup[target]
                ctxid = space.lookup[context]
                relid = rels[relation]
            except KeyError:
                continue
            if tarid is not last:
                if last:
                    yield last, cum
                cum = []
                last = tarid
            cum.append((relid, ctxid))

def main():
    parser = argparse.ArgumentParser('description')
    parser.add_argument('--input', '-i', default='-', help='Input corpus')
    parser.add_argument('--output', '-o', help='Output numpy file')
    parser.add_argument('--relations', '-r', help='Relations file')
    parser.add_argument('--space', '-s', help='Space filename')
    parser.add_argument('--mindeps', '-m', type=int, default=1,
                        help='Minimum number of attachments to store in matrix.')
    parser.add_argument('--maxrels', '-M', type=int, default=1000,
                        help='Maximum number of relationships to model.')
    args = parser.parse_args()

    space = load_numpy(args.space, insertblank=True)
    rels = read_relationships(args.relations, args.maxrels)

    targetids = []

    rowids = []
    colids = []
    datavals = []

    num_rows = 0
    num_skipped = 0
    num_overlap = 0
    num_rows_with_overlap = 0

    out_counter = 0

    rowid = 0
    for targetid, relcontexts in pull_items(args.input, space, rels):
        relcontexts_d = {}
        for rel, ctx in relcontexts:
            relcontexts_d[rel] = max(relcontexts_d.get(rel, 0), ctx)

        if len(relcontexts_d) < args.mindeps:
            num_skipped += 1
            continue

        num_rows += 1
        overlap = len(relcontexts) - len(relcontexts_d)
        if overlap:
            num_overlap += overlap
            num_rows_with_overlap += 1

        for rel, ctx in relcontexts_d.iteritems():
            rowids.append(rowid)
            colids.append(rel)
            datavals.append(ctx)
        targetids.append(targetid)
        rowid += 1

        # magic number means ~25MB output files, while being able
        # to be broken into nice 128 row chunks
        # This is important so that we can keep the memory usage down low later
        if rowid >= 2097152:
            print "\nSaving chunk %06d" % out_counter
            targetoutputs = np.array(targetids, dtype=np.int32)
            output = scipy.sparse.csr_matrix((datavals, (rowids, colids)), dtype=np.int32)
            outputname = "%s/chunk_%04d.npz" % (args.output, out_counter)
            np.savez_compressed(outputname, targets=targetoutputs, contexts=output)
            del targetoutputs
            del output
            rowid = 0
            targetids = []
            rowids = []
            colids = []
            datavals = []
            out_counter += 1

    if targetids:
        targetoutputs = np.array(targetids)
        output = scipy.sparse.csr_matrix((datavals, (rowids, colids)), dtype=np.int32)
        outputname = "%s/chunk_%06d.npz" % (args.output, out_counter)
        np.savez_compressed(outputname, targets=targetoutputs, contexts=output)

    print "Number of accepted rows:", num_rows
    print " Number of skipped rows:", num_skipped
    print "  Number of overlapping:", num_overlap
    print "  Number of rows w/ ovr:", num_rows_with_overlap


if __name__ == '__main__':
    main()

