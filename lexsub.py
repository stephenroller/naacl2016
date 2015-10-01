#!/usr/bin/env python
import sys
import os.path
import argparse
import numpy as np
from itertools import izip
from collections import defaultdict
from sklearn.preprocessing import normalize

import depextract
import utdeftvs

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
                target = rewrite_pos(target)
                ident = int(ident)
                rights = [r for r in right.split(";") if r]
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
        self.tokens = [tokens[k] for k in idents]
        self.targets = [targets[k] for k in idents]
        self.parses = [parses[k] for k in idents]

        self.golds = defaultdict(dict)
        for ident, subs in golds.iteritems():
            for c in candidates[targets[ident]]:
                self.golds[ident][c] = subs.get(c, 0)
        self.golds = [self.golds[k] for k in idents]

        assert len(self.targets) == len(self.parses) == len(self.golds)


    def generate_matrices(self, vocablookup):
        targets = np.zeros(len(self.targets), dtype=np.int32)
        numtargets = len(self.targets)
        maxcands = max(len(g) for g in self.golds)

        # the sister matrices
        subs = np.zeros((numtargets, maxcands), dtype=np.int32)
        scores = np.zeros((numtargets, maxcands), dtype=np.float32)

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
        self.tokens = np.array(self.tokens)[targets != 0]
        self.parses = np.array(self.parses)[targets != 0]
        scores = scores[targets != 0]
        targets = targets[targets != 0]

        return targets, subs, scores

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
                    print '-', dep

    numrows = len(deps)
    numcols = max(len(d) for d in deps)
    depmat = np.zeros((numrows, numcols), dtype=np.int32)

    for i, d in enumerate(deps):
        l = len(d)
        depmat[i,:l] = d

    return depmat


def compute_oren(space, targets, depmat, candidates):
    normspace = space.normalize()

    targetvecs = normspace.matrix[targets] # (2003, 600)
    depvecs = normspace.cmatrix[depmat] # (2003, 14, 600)
    n = (depmat > 0).sum(axis=1) + 1 # (2003,)
    candvecs = normspace.matrix[candidates] # (2003, 38, 600)

    left = np.einsum('ij,ikj->ik', targetvecs, candvecs)
    right = np.einsum('ijk,ilk->il', depvecs, candvecs)
    pred_scores = (left + right) / n[:,np.newaxis]

    return pred_scores

def compute_fakeoren(space, targets, depmat, candidates):
    normspace = space.normalize()
    targetvecs = normspace.matrix[targets]
    candvecs = normspace.matrix[candidates]
    depvecs = normspace.cmatrix[depmat].sum(axis=1)

    # dist = || (a - p) - ((a - p)^Tn)n ||
    n = normalize(depvecs - targetvecs, axis=1, norm='l2') # (1972, 300)
    a_p = targetvecs[:,np.newaxis,:] - candvecs # (1972, 38, 300)
    a_pTn = np.einsum('ijk,ik->ij', a_p, n) # (1972, 38)
    a_pTnn = np.multiply(a_pTn[:,:,np.newaxis], n[:,np.newaxis,:]) # (1972, 38, 300)

    dists = np.sqrt(np.sum(np.square(a_p - a_pTnn), axis=2))
    return -dists


def compute_ooc(space, targets, candidates):
    normspace = space.normalize()
    #normspace = space
    targetvecs = normspace.matrix[targets]
    candvecs = normspace.matrix[candidates]
    pred_scores = np.einsum('ij,ikj->ik', targetvecs, candvecs)
    return pred_scores

def compute_random(candidates):
    return np.random.rand(*candidates.shape)

def compute_oracle(candidates, scores, space):
    normspace = space.normalize()
    perfect = np.multiply(space.matrix[candidates], scores[:,:,np.newaxis]).sum(axis=1)
    perfect = normalize(perfect, axis=1, norm='l2')
    pred_scores = np.einsum('ij,ikj->ik', perfect, normspace.matrix[candidates])
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



def main():
    parser = argparse.ArgumentParser('Performs lexical substitution')
    args = parser.parse_args()

    # load the data
    #semeval = LexsubData("data/semeval_all")
    semeval = LexsubData("data/coinco")
    # load the space
    #space = utdeftvs.load_numpy("/scratch/cluster/roller/spaces/giga+bnc+uk+wiki2015/output/dependency.svd300.ppmi.250k.1m.npz", True)
    #space = utdeftvs.load_numpy("/scratch/cluster/roller/spaces/giga+bnc+uk+wiki2015/dependency/output/dependency.w2v500.top250k.top1m.npz", True)
    space = utdeftvs.load_numpy("/scratch/cluster/roller/spaces/levy/lexsub_embeddings.npz", True)
    # need to map our vocabulary to their indices
    targets, candidates, scores = semeval.generate_matrices(space.lookup)
    depmat = dependencies_to_indices(semeval.tokens, semeval.parses, space.clookup, space)
    print depmat

    #pred_scores = compute_fakeoren(space, targets, depmat, candidates)
    #pred_scores = compute_oren(space, targets, depmat, candidates)
    pred_scores = compute_ooc(space, targets, candidates)
    #pred_scores = compute_random(candidates)
    #pred_scores = compute_oracle(candidates, scores, space)
    assert scores.shape == pred_scores.shape

    #pred_scores = pred_scores[2:3,:]
    #scores = scores[2:3,:]

    gaps = []
    for i in xrange(pred_scores.shape[0]):
        cands = candidates[i]
        gw = scores[i]
        gold_indices = cands[gw > 0]
        gold_weights = gw[gw > 0]
        if np.all(gw == 0):
            continue
        ranks = (-pred_scores[i]).argsort()
        ranked_candidate_indices = cands[ranks]
        ranked_candidate_indices = ranked_candidate_indices[ranked_candidate_indices != 0]
        gaps.append(gap_correct(gold_indices, gold_weights, ranked_candidate_indices))
    gaps = np.array(gaps)

    #mygap = gap(pred_scores, scores)
    #print mygap
    gaps = gaps[~np.isnan(gaps)]
    print gaps
    print np.mean(gaps)
    #print np.mean(mygap)




if __name__ == '__main__':
    main()
