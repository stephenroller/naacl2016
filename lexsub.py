#!/usr/bin/env python
import sys
import os.path
import argparse
import numpy as np
import random
from unidecode import unidecode
from itertools import izip
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import normalize

import depextract
import utdeftvs
import ctxpredict.models

from nn import *

USE_LEMMAPOS = False

def splitpop(string, delimeter):
    """
    Splits a string along a delimiter, and returns the
    string without the last field, and the last field.

    >>> splitpop('hello.world.test', '.')
    'hello.world', 'test'
    """
    fields = string.split(delimeter)
    return delimeter.join(fields[:-1]), fields[-1]

def find_start(string, index):
    words = string.split(" ")
    before = words[:index]
    if not before:
        return 0
    return len(" ".join(before)) + 1

def revsorted(arr):
    random.shuffle(arr)
    return sorted(arr, key=lambda x: x[1], reverse=True)

def rewrite_pos(string):
    # stupid data exceptions :(
    if string == '.':
        return string
    elif string == '..N':
        return '.'
    word, pos = splitpop(string, ".")
    if '.' in word:
        word, pos = splitpop(word, ".")
    word = word.lower()
    pos = pos.lower()
    if pos == 'n':
        return word + '/NN'
    elif pos == 'j' or pos == 'a':
        return word + '/JJ'
    elif pos == 'v':
        return word + '/VB'
    elif pos == 'r':
        return word + '/RB'
    else:
        raise ValueError("Don't know how to handle a POS tag of '%s' in '%s'" % (pos, string))

def scrub_substitutes(before, target):
    """
    Not all substitutes are good substitutes. We need to clean them up,
    lemmatize them, add POS tags, and remove multiword expressions.

    Input: a dictionary with uncleaned substitute/weight as key/value
    Returns: a dictionary with cleaned subsititute/weight as key/value

    This dictionary will (probably) be smaller than the input dictionary.
    """
    targetpos = USE_LEMMAPOS and target[-3:] or ""
    before_iter = before.iteritems()
    remove_mwe = ((k, v) for k, v in before_iter if ' ' not in k)
    remove_dash = ((k, v) for k, v in remove_mwe if '-' not in k)
    add_pos = ((k + targetpos, v) for k, v in remove_mwe)
    return dict(add_pos)

def scrub_candidates(before, target):
    fakesubs = {k: 1 for k in before}
    return scrub_substitutes(fakesubs, target).keys()

class LexsubData(object):
    def __init__(self, foldername):
        # load gold file
        golds = {}
        targets = {}
        original_targets = {}
        candidates = {}
        sentences = defaultdict(list)
        starts = defaultdict(list)
        tokens = {}

        with open(os.path.join(foldername, "gold")) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if "::" not in line:
                    continue
                left, right = line.split("::")
                left = left.strip()
                right = right.strip()
                target, ident = splitpop(left, " ")
                ident = int(ident)
                original_targets[ident] = target
                target = rewrite_pos(target)
                rights = [r.strip() for r in right.split(";") if r.strip()]
                # gold_golds is what the data explicitly says are the gold substitutes
                # but these are not lemmatized or POS tagged, and contain MWE. We
                # need to clean them up with scrub_substitutes
                gold_golds = {k: int(v) for k, v in (splitpop(r, ' ') for r in rights)}
                golds[ident] = scrub_substitutes(gold_golds, target)
                targets[ident] = target

        with open(os.path.join(foldername, "sentences")) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if "\t" not in line:
                    continue
                target, ident, index, sentence = line.split("\t")
                target = rewrite_pos(target)
                ident = int(ident)
                index = int(index)
                if ident not in targets:
                    continue
                if targets[ident] != target:
                    raise ValueError("Something didn't line up: found %s, expected %s" % (target, targets[ident]))
                sentences[sentence].append(ident)
                starts[ident] = find_start(sentence, index)

        # parse the sentences
        plaintexts = sentences.keys()
        parsed_sentences = depextract.preprocess_with_corenlp(os.path.join(foldername, "parses"), plaintexts)
        parsed_sentences = depextract.parse_corenlp_xml(parsed_sentences, dependencytype='basic-dependencies')
        parses = {}
        for sentence, parse in izip(sentences.iterkeys(), parsed_sentences):
            idents = sentences[sentence]
            for ident in idents:
                parses[ident] = parse
                for t in parse.tokens:
                    if t.start == starts[ident]:
                        tokens[ident] = t
                        break
                else:
                    raise IndexError("These are not the words you are looking for.")


        ## generate candidates
        #candidates = defaultdict(set)
        #for ident, subs in golds.iteritems():
        #    candidates[targets[ident]].update(subs.keys())
        candidates = {}
        with open(os.path.join(foldername, "candidates")) as f:
            for line in f:
                line = line.strip()
                left, right = line.split("::")
                target = rewrite_pos(left)
                candidates[target] = scrub_candidates(right.split(";"), target)

        idents = targets.keys()
        self.idents = idents
        self.tokens = [tokens[k] for k in idents]
        self.original_targets = [original_targets[k] for k in idents]
        self.targets = [targets[k] for k in idents]
        self.parses = [parses[k] for k in idents]

        self.golds = defaultdict(dict)
        for ident, subs in golds.iteritems():
            for c in candidates[targets[ident]]:
                self.golds[ident][c] = subs.get(c, 0)
        self.golds = [self.golds[k] for k in idents]

        assert len(self.targets) == len(self.parses) == len(self.golds) == len(self.idents) == len(self.original_targets)


    def generate_matrices(self, vocablookup):
        targets = np.zeros(len(self.targets), dtype=np.int32)
        numtargets = len(self.targets)
        maxcands = max(len(g) for g in self.golds)

        # the sister matrices
        subs = np.zeros((numtargets, maxcands), dtype=np.int32)
        scores = np.zeros((numtargets, maxcands), dtype=np.float32)

        targets_with_pos = []

        # we want to produce a a matrix which contains one target per row
        # each column will have the ID of the substitute and the corresponding
        # number of substitutions in its sister matrix
        # but we want them to be ordered so the ID with the most subs is first
        for i, (token, gold) in enumerate(izip(self.tokens, self.golds)):
            target = token.word_normed #USE_LEMMAPOS and token.lemma_pos or token.word_normed
            idx = vocablookup.get(target, 0)
            targets[i] = idx
            # we want items not in our vocab to have a 0 weight
            orderedgold = sorted(gold.keys(), key=lambda x: gold[x] or 0, reverse=True)
            j = 0
            for g in orderedgold:
                if g not in vocablookup:
                    continue
                subs[i,j] = vocablookup.get(g, 0)
                scores[i,j] = gold[g]
                j += 1

        # get rid of data points where the target was OOV
        subs = subs[targets != 0]
        self.targets = np.array(self.targets)[targets != 0]
        self.idents = np.array(self.idents)[targets != 0]
        self.tokens = np.array(self.tokens)[targets != 0]
        self.parses = np.array(self.parses)[targets != 0]
        scores = scores[targets != 0]
        targets = targets[targets != 0]

        assert len(self.targets) == len(self.parses) == len(self.idents)

        return self.idents, targets, subs, scores

def dependencies_to_indices(target_tokens, parses, lookup,space):
    deps = []
    for target, parse in izip(target_tokens, parses):
        deps.append([])
        if USE_LEMMAPOS:
            extractor = depextract.extract_relations_for_token(parse, target)
        else:
            extractor = depextract.extract_relations_for_token_melamud(parse, target, inverter='I')
        for relation, attachment, in extractor:
            if USE_LEMMAPOS:
                dep = relation + "+" + attachment.lemma_pos
            else:
                dep = relation + "_" + attachment.word_normed
            if dep in lookup:
                deps[-1].append(lookup[dep])
            else:
                if attachment.word_normed in space.lookup:
                    #print '-', dep
                    pass

    numrows = len(deps)
    numcols = max(len(d) for d in deps)
    depmat = np.zeros((numrows, numcols), dtype=np.int32)

    for i, d in enumerate(deps):
        l = len(d)
        depmat[i,:l] = d

    return depmat[:,::-1]

def dependencies_to_indicies3(target_tokens, parses, vlookup, rlookup):
    deps = []
    rels = []
    for target, parse in izip(target_tokens, parses):
        d = np.zeros(10)
        r = np.zeros(10)
        i = 0
        relattachments = list(depextract.extract_relations_for_token_melamud(parse, target, inverter='I'))
        for relation, attachment in relattachments:
            if i >= 10:
                break
            if relation not in rlookup or attachment.word_normed not in vlookup:
                continue
            rid = rlookup[relation] + 1
            vid = vlookup[attachment.word_normed]
            d[i] = vid
            r[i] = rid
            i += 1
        deps.append(d)
        rels.append(r)
    return [np.array(deps), np.array(rels)]

def read_relationships(filename, max_relationships):
    labels = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i >= max_relationships:
                break
            labels.append(line.strip().split("\t")[0])
    return {l: i for i, l in enumerate(labels)}

def magnitude(x):
    return np.sqrt(np.sum(np.square(x), axis=-1))

def fix_pred_scores(pred_scores, candidates):
    r = pred_scores.copy()
    r[candidates == 0] = -1e9
    return r

def fix_allvocab_pred_scores(pred_scores, targets):
    r = pred_scores.copy()
    r[:,0] = -1e9
    for i, t in enumerate(targets):
        r[i,t] = -1e9
    return r

def compute_mymodel(space, targets, model, depmat, candidates, batch_size=1024):
    predvecs = model.predict([depmat, candidates], batch_size=batch_size)
    predvecs[candidates == 0] = 0
    predvecs = normalize(predvecs, norm='l1')
    targetvecs = space.matrix[targets] # (2003, 600)
    candvecs = space.matrix[candidates]
    ooc = np.exp(np.einsum('ij,ikj->ik', targetvecs, candvecs))
    ooc = normalize(ooc, norm='l1', axis=1)
    scores = np.log(predvecs) + np.log(ooc)
    return scores

def compute_mymodel_allwords(space, targets, model, depmat):
    candidates = np.repeat([np.arange(len(space.vocab))], len(targets), axis=0)
    predvecs = model.predict([depmat, candidates], batch_size=8)
    predvecs = np.log(normalize(predvecs, norm='l1'))
    targetvecs = space.matrix[targets]
    ooc = np.exp(np.dot(targetvecs, space.matrix.T))
    ooc = np.log(normalize(ooc, norm='l1', axis=1))
    scores = predvecs + ooc
    return scores

def compute_oren(space, targets, depmat, candidates):
    normspace = space.normalize()

    targetvecs = normspace.matrix[targets] # (2003, 600)
    depvecs = normspace.cmatrix[depmat] # (2003, 14, 600)
    candvecs = normspace.matrix[candidates] # (2003, 38, 600)

    left = np.einsum('ij,ikj->ik', targetvecs, candvecs)
    right = np.einsum('ijk,ilk->il', depvecs, candvecs)
    pred_scores = (left + right)

    return pred_scores

def compute_oren_allvocab(space, targets, depmat):
    normspace = space.normalize()

    targetvecs = normspace.matrix[targets]
    depvecs = normspace.cmatrix[depmat]
    left = targetvecs.dot(normspace.matrix.T)
    right = depvecs.sum(axis=1).dot(normspace.matrix.T)
    pred_scores = (left + right)
    return pred_scores

def compute_ooc(space, targets, candidates):
    normspace = space.normalize()
    targetvecs = normspace.matrix[targets]
    candvecs = normspace.matrix[candidates]
    pred_scores = np.einsum('ij,ikj->ik', targetvecs, candvecs)
    return pred_scores

def compute_ooc_allvocab(space, targets):
    normspace = space.normalize()
    targetvecs = normspace.matrix[targets]
    pred_scores = targetvecs.dot(normspace.matrix.T)
    return pred_scores

def compute_random(candidates):
    return np.random.rand(*candidates.shape)

def compute_oracle(candidates, scores, space):
    normspace = space.normalize()
    perfect = np.multiply(space.matrix[candidates], scores[:,:,np.newaxis]).sum(axis=1)
    perfect = normalize(perfect, axis=1, norm='l2')
    pred_scores = np.einsum('ij,ikj->ik', perfect, normspace.matrix[candidates])
    return pred_scores

def compute_oracle_allvocab(scores, space):
    normspace = space.normalize
    perfect = normalize(scores.dot(space.matrix))
    pred_scores = perfect.dot(normspace.matrix.T)
    return pred_scores



#def gap(pred_scores, true_scores):
#    maxcands = pred_scores.shape[1]
#    maxi = np.sum(true_scores > 0, axis=1) - 1
#
#    pred_scores = pred_scores[maxi >= 0]
#    true_scores = true_scores[maxi >= 0]
#    maxi = maxi[maxi >= 0]
#
#    ranks = (-pred_scores).argsort(axis=1)
#    points_by_pred = np.array([ts[r] for r, ts in izip(ranks, true_scores)])
#
#    bot = np.arange(maxcands) + 1
#    max_points = true_scores.cumsum(axis=1) / bot[np.newaxis,:]
#    actual_points = points_by_pred.cumsum(axis=1) / bot[np.newaxis,:]
#    numers = np.array([np.sum(ap[pbp > 0]) for ap, pbp in izip(actual_points, points_by_pred)])
#    denoms = np.array([np.sum(mp[:i+1]) for i, mp in izip(maxi, max_points)])
#
#    gaps = numers / denoms
#
#    return gaps

def gap_correct(gold_indices, gold_weights, ranked_candidate_indices):
    # generalized average precision
    gold_reverse_lookup = {id : i for i, id in enumerate(gold_indices)}
    cumsum = 0.0
    gap = 0.0
    for i, id in enumerate(ranked_candidate_indices):
        if id not in gold_reverse_lookup:
            continue
        subj = gold_reverse_lookup[id]
        cumsum += gold_weights[subj]
        gap += cumsum / (i + 1)
        #print "i = %f, cumsum = %f, gap = %f" % (i, cumsum, gap)

    cumsum = 0.0
    denom = 0.0
    for i, weight in enumerate(gold_weights):
        #cumsum += weight
        #denom += cumsum / (i + 1)
        denom += np.mean(gold_weights[:i+1])
    #print "final num = %f, final denom = %f" % (gap, denom) 
    gap = gap / denom

    return gap

def many_gaps(pred_scores, candidates, scores):
    assert scores.shape == pred_scores.shape
    gaps = []
    for i in xrange(pred_scores.shape[0]):
        cands = candidates[i]
        gw = scores[i]
        gold_indices = cands[gw > 0]
        gold_weights = gw[gw > 0]
        if np.all(gw == 0):
            gaps.append(np.nan)
            continue
        ranks = (-pred_scores[i]).argsort()
        ranked_candidate_indices = cands[ranks]
        ranked_candidate_indices = ranked_candidate_indices[ranked_candidate_indices != 0]
        gaps.append(gap_correct(gold_indices, gold_weights, ranked_candidate_indices))
    return np.array(gaps)

def nanmean(arr):
    return np.mean(arr[~np.isnan(arr)])

def many_prec1(pred_scores, scores):
    assert scores.shape == pred_scores.shape
    bestpick = pred_scores.argmax(axis=1)
    score_at_best = scores[np.arange(len(scores)),bestpick]
    gold_values = (scores.max(axis=1) > 0)
    retval = np.array(score_at_best > 0, dtype=np.float) / gold_values
    return retval

def prec_at_k(pred_scores, scores, k=10, mask=None):
    top_preds = np.argpartition(pred_scores, -k, 1)[:,-k:]
    num_possible = (scores > 0).sum(axis=1, dtype=np.float).clip(0, k)

    top_pred_true = np.array([s[tp] for s, tp in izip(scores, top_preds)])
    num_right = (top_pred_true > 0).sum(axis=1, dtype=np.float)
    num_right[num_possible == 0] = np.nan
    return num_right / k

from common import *
from nn import my_load_weights

def main():
    parser = argparse.ArgumentParser('Performs lexical substitution')
    parser.add_argument('--model', '-m')
    parser.add_argument('--data', '-d')
    parser.add_argument('--allvocab', action='store_true')
    parser.add_argument('--baseline', choices=('oren', 'random', 'ooc', 'oracle', 'orensm', 'orenun'))
    parser.add_argument('--save')
    parser.add_argument('--semeval')
    args = parser.parse_args()

    if (args.model and args.baseline) or (not args.model and not args.baseline):
        raise ValueError("Please supply exactly one of model or baseline.")
    if args.semeval and not args.allvocab:
        raise ValueError("Need to evaluate on allvocab to output semeval predictions.")

    if not args.data:
        raise ValueError("You must specify a data folder")

    # load the data
    semeval = LexsubData(args.data)
    space = utdeftvs.load_numpy("/work/01813/roller/maverick/nnexp/lexsub_embeddings.npz", True)
    #relations = read_relationships("/work/01813/roller/maverick/nnexp/relations.txt", 1000)
    #model = ctxpredict.models.get_model("2d", space, len(relations), space.matrix.shape[1])
    # load the space
    #space = utdeftvs.load_numpy("/scratch/cluster/roller/spaces/giga+bnc+uk+wiki2015/output/dependency.svd300.ppmi.250k.1m.npz", True)
    #space = utdeftvs.load_numpy("/scratch/cluster/roller/spaces/giga+bnc+uk+wiki2015/dependency/output/dependency.w2v500.top250k.top1m.npz", True)
    #space = utdeftvs.load_numpy("/scratch/cluster/roller/spaces/levy/lexsub_embeddings.npz", True)
    # need to map our vocabulary to their indices
    ids, targets, candidates, scores = semeval.generate_matrices(space.lookup)
    depmat = dependencies_to_indices(semeval.tokens, semeval.parses, space.clookup, space)
    print "Done preprocessing"


    if args.allvocab:
        allvocab_scores = np.zeros((len(targets), len(space.vocab)))
        for i in xrange(len(targets)):
            for j in xrange(candidates.shape[1]):
                c = candidates[i,j]
                s = scores[i,j]
                if s > 0:
                    allvocab_scores[i,c] = s
    allvocab_pred_scores = np.zeros(len(space.vocab))
    if args.baseline:
        print "Computing baseline %s" % args.baseline
        if args.baseline == 'oren':
            pred_scores = compute_oren(space, targets, depmat, candidates)
            if args.allvocab:
                allvocab_pred_scores = compute_oren_allvocab(space, targets, depmat)
        elif args.baseline == 'ooc':
            pred_scores = compute_ooc(space, targets, candidates)
            if args.allvocab:
                allvocab_pred_scores = compute_ooc_allvocab(space, targets)
        elif args.baseline == 'random':
            pred_scores = compute_random(candidates)
        elif args.baseline == 'oracle':
            pred_scores = compute_oracle(candidates, scores, space)
        else:
            pred_scores = np.zeros(candidates.shape)
        modelname = "baseline"
        modelinfo = args.baseline
    elif args.model:
        model = my_model_from_json(args.model + "/model.json")
        filename = sorted(os.listdir(args.model))[-1]
        my_load_weights(model, "%s/%s" % (args.model, filename))

        pred_scores = compute_mymodel(space, targets, model, depmat, candidates)
        pred_scores = fix_pred_scores(pred_scores, candidates)
        if args.allvocab:
            allvocab_pred_scores = compute_mymodel_allwords(space, targets, model, depmat)
        modelname = args.model
        modelinfo = filename
    else:
        raise ValueError("Not given model or baseline to compute...")

    # make sure we're not guessing the target, or the empty vector
    pred_scores = fix_pred_scores(pred_scores, candidates)
    if args.allvocab:
        allvocab_pred_scores = fix_allvocab_pred_scores(allvocab_pred_scores, targets)

    # compute evaluations
    gaps = many_gaps(pred_scores, candidates, scores)
    prec1s = many_prec1(pred_scores, scores)
    prec3s = prec_at_k(pred_scores, scores, 3)
    # allvocab is slow; only compute that if we have to
    if args.allvocab:
        prec1s_av = many_prec1(allvocab_pred_scores, allvocab_scores)
        prec3s_av = prec_at_k(allvocab_pred_scores, allvocab_scores, 3)
    else:
        prec1s_av = np.zeros(len(targets))
        prec3s_av = np.zeros(len(targets))

    print ("%s\t%s\t%s\tgap %.3f\tp@1 %.3f\tp@3 %.3f\tp@1av %.3f\tp@3av %.3f" %
            (args.data, modelname, modelinfo,
             nanmean(gaps), nanmean(prec1s), nanmean(prec3s), nanmean(prec1s_av), nanmean(prec3s_av)))

    if args.save:
        with open(args.save, 'w') as f:
            f.write('\t'.join(['ident', 'target', 'sentence', 'gold', 'predicted', 'gap', 'p@1', 'p@3', 'p@1av', 'p@3av']))
            f.write('\n')
            for i in xrange(len(semeval.idents)):
                ident = semeval.idents[i]
                target = semeval.targets[i]
                parse = semeval.parses[i]
                scores_i = scores[i]
                pred_scores_i = pred_scores[i]
                candidates_i = candidates[i]
                gap = gaps[i]
                prec1 = prec1s[i]
                prec3 = prec3s[i]
                prec1av = prec1s_av[i]
                prec3av = prec3s_av[i]

                sentence = " ".join(t.word_normed for t in parse.tokens)

                score_string = " ".join("%s:%3.1f" % (space.vocab[c], s) for c, s in zip(candidates_i, scores_i) if c != 0)
                pred_string = " ".join("%s:%f" % (space.vocab[c], p) for c, p in revsorted(zip(candidates_i, pred_scores_i)) if c != 0)
                outline = '\t'.join([str(ident), target, sentence, score_string, pred_string, str(gap), str(prec1), str(prec3), str(prec1av), str(prec3av)])
                outline = unidecode(outline)
                f.write(outline)
                f.write('\n')
    if args.semeval:
        bestf = open(args.semeval + ".best", "w")
        bootf = open(args.semeval + ".boot", "w")
        bests = allvocab_pred_scores.argmax(axis=1)
        boots = np.argpartition(allvocab_pred_scores, -10, 1)[:,-10:]
        for i in xrange(len(semeval.idents)):
            ident = semeval.idents[i]
            ot = semeval.original_targets[i]
            bestf.write("%s %d :: %s\n" % (ot, ident, space.vocab[bests[i]]))
            bootf.write("%s %d ::: %s\n" % (ot, ident, ";".join(space.vocab[boots[i]])))
        bestf.close()
        bootf.close()



if __name__ == '__main__':
    main()

