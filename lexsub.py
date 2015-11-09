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
import ctxpredict.models

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
                    #print '-', dep
                    pass

    numrows = len(deps)
    numcols = max(len(d) for d in deps)
    depmat = np.zeros((numrows, numcols), dtype=np.int32)

    for i, d in enumerate(deps):
        l = len(d)
        depmat[i,:l] = d

    return depmat

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


def compute_mymodel(space, targets, model, depmat, candidates):
    #model.train_on_batch(depmat, targets)
    predvecs = model.predict(depmat, batch_size=1024) #+ space.matrix[targets]
    predvecs = normalize(predvecs, norm='l2', axis=1)
    normspace = space.normalize()
    candvecs = space.matrix[candidates]
    targetvecs = normspace.matrix[targets] # (2003, 600)
    predvecs += targetvecs
    scores = np.einsum('ij,ikj->ik', predvecs, candvecs)

    return scores


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

def many_gaps(pred_scores, candidates, scores):
    assert scores.shape == pred_scores.shape
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
    gaps = gaps[~np.isnan(gaps)]
    return gaps

from common import *
from nn import my_load_weights

def main():
    parser = argparse.ArgumentParser('Performs lexical substitution')
    parser.add_argument('--model', '-m')
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--data', '-d')
    args = parser.parse_args()

    if not args.data:
        raise ValueError("You must specify a data folder")

    # load the data
    semeval = LexsubData(args.data)
    space = utdeftvs.load_numpy("/work/01813/roller/maverick/nnexp/lexsub_embeddings.npz", True)
    relations = read_relationships("/work/01813/roller/maverick/nnexp/relations.txt", 1000)
    #model = ctxpredict.models.get_model("2d", space, len(relations), space.matrix.shape[1])
    # load the space
    #space = utdeftvs.load_numpy("/scratch/cluster/roller/spaces/giga+bnc+uk+wiki2015/output/dependency.svd300.ppmi.250k.1m.npz", True)
    #space = utdeftvs.load_numpy("/scratch/cluster/roller/spaces/giga+bnc+uk+wiki2015/dependency/output/dependency.w2v500.top250k.top1m.npz", True)
    #space = utdeftvs.load_numpy("/scratch/cluster/roller/spaces/levy/lexsub_embeddings.npz", True)
    # need to map our vocabulary to their indices
    targets, candidates, scores = semeval.generate_matrices(space.lookup)
    #depmat = dependencies_to_indices(semeval.tokens, semeval.parses, space.clookup, space)
    depmat = dependencies_to_indicies3(semeval.tokens, semeval.parses, space.lookup, relations)
    # converting the data set to be supervised:
    if args.supervise:
        print "Generating supervised variation. Repeating observed targets %d times." % int(scores.max())
        original_target = []
        super_y = []
        super_contexts = []
        super_rels = []
        for i in xrange(len(targets)):
            for j in xrange(int(scores.max())):
                break
                super_y.append(targets[i])
                super_contexts.append(depmat[0][i])
                super_rels.append(depmat[1][i])
                original_target.append(targets[i])
            for c, s in zip(candidates[i], scores[i]):
                if s <= 0: continue
                for j in xrange(int(s)):
                    super_y.append(c)
                    super_contexts.append(depmat[0][i])
                    super_rels.append(depmat[1][i])
                    original_target.append(targets[i])
        original_target = np.array(original_target)
        super_contexts = np.array(super_contexts)
        super_rels = np.array(super_rels)
        super_y = np.array(super_y)

        np.savez_compressed("%s/data.npz" % args.data, original_target, super_y, super_contexts, super_rels)

        ot = set(original_target)
        sample = set(np.random.choice(list(ot), len(ot)/10, False))
        train = np.array([y not in sample for y in original_target])
        train_bits = np.array([t not in sample for t in targets])
        assert len(set(original_target[train]).intersection(set(targets[~train_bits]))) == 0
        print "Result of supervised generation: %d items." % len(super_y)

    espace = utdeftvs.VectorSpace(embedding_matrix, space.vocab)
    #import ipdb; ipdb.set_trace()
    #print depmat

    MODEL_FOLDER = args.model
    import keras.models
    model = keras.models.model_from_json(open(MODEL_FOLDER + "/model.json").read())
    for filename in sorted(os.listdir(MODEL_FOLDER))[-1:]:
        if not filename.endswith('.npz'):
            continue
        my_load_weights(model, embedding_matrix, "%s/%s" % (MODEL_FOLDER, filename))
        model.optimizer.lr = 0.01

        if args.supervise:
            baseline = model.evaluate([super_contexts[~train], super_rels[~train]], embedding_matrix[super_y[~train]])
            pred_scores = compute_mymodel(espace, targets[~train_bits], model, 
                    [depmat[0][~train_bits], depmat[1][~train_bits]], candidates[~train_bits])
            gaps = many_gaps(pred_scores, candidates[~train_bits], scores[~train_bits])
            print "Before training: %6.4f    %.4f" % (baseline, np.mean(gaps))
            print model.optimizer.lr
            for x in xrange(10):
                model.fit([super_contexts[train], super_rels[train]], embedding_matrix[super_y[train]],
                        validation_data=([super_contexts[~train], super_rels[~train]], embedding_matrix[super_y[~train]]),
                        verbose=True, nb_epoch=1, batch_size=1024)
                pred_scores = compute_mymodel(espace, targets[~train_bits], model, 
                        [depmat[0][~train_bits], depmat[1][~train_bits]], candidates[~train_bits])
                gaps = many_gaps(pred_scores, candidates[~train_bits], scores[~train_bits])
                print "After_training\t%s\t%s\t%.4f" % (MODEL_FOLDER, filename, np.mean(gaps))
        else:
            pred_scores = compute_mymodel(espace, targets, model, depmat, candidates)
            gaps = many_gaps(pred_scores, candidates, scores)
            print "Unsupervised\t%s\t%s\tgap\t%.4f" % (MODEL_FOLDER, filename, np.mean(gaps))


        #pred_scores = compute_mymodel(space, targets, model, depmat, candidates)

        #pred_scores = compute_mymodel(espace, targets[~train_bits], model, depmat[~train_bits], candidates[~train_bits])
        #pred_scores = compute_oren(space, targets, depmat, candidates)
        #pred_scores = compute_ooc(space, targets, candidates)
        #pred_scores = compute_random(candidates)
        #pred_scores = compute_oracle(candidates, scores, space)

        #pred_scores = pred_scores[2:3,:]
        #scores = scores[2:3,:]


        #print gaps
        #print np.mean(mygap)




if __name__ == '__main__':
    main()
