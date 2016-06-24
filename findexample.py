#!/usr/bin/env python

import numpy as np
import pandas as pd
from itertools import izip

def read_semeval(fname):
    output = []
    idents = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            left, right = line.split(" ::: ")
            word, ident = left.split(" ")
            ident = int(ident)
            subs = right.split(";")
            row = {
                'target': word,
            }
            idents.append(ident)
            for i, s in enumerate(subs):
                row["sub%02d" % (i + 1)] = s
            output.append(row)
    return pd.DataFrame(output, index=idents)

ooc = pd.read_table("recompute_predictions/ooc.tsv")
ooc_p = read_semeval("recompute_predictions/se_ooc.boot")
baloren = pd.read_table("recompute_predictions/baloren.tsv")
baloren_p = read_semeval("recompute_predictions/se_baloren.boot")
orensm = pd.read_table("recompute_predictions/orensm.tsv")
orensm_p = read_semeval("recompute_predictions/se_orensm.boot")
pic = pd.read_table("recompute_predictions/pic.tsv")
pic_p = read_semeval("recompute_predictions/pic.boot")

mask_cherry = (baloren['p@3av'] < orensm['p@3av']) & (orensm['p@1av'] < pic['p@1av'])
mask_lemon = (baloren['p@3av'] > orensm['p@3av']) & (pic['p@1av'] == 0.0) & (baloren['p@1av'] == 1.0)

#mask = mask_cherry | mask_lemon
#mask = mask_lemon
mask = mask_cherry

sel_baloren = baloren[mask].iterrows()
sel_orensm = orensm[mask].iterrows()
sel_pic = pic[mask].iterrows()

tables = (ooc_p, baloren_p, orensm_p, pic_p)

def bf(yes):
    if yes:
        return '\\bf '
    else:
        return '    '

for (b, brow), (o, orow), (u, row) in izip(sel_baloren, sel_orensm, sel_pic):
    assert (brow['ident'] == orow['ident'] == row['ident'])
    print "ID: %d" % brow['ident']
    print "Target: %s" % row['target']
    word = row['target'][:row['target'].rindex('/')]
    print "Context: %s" % row['sentence']
    golds = row['gold'].split(' ')
    golds = [(k, float(v)) for k, v in [g.split(':') for g in golds]]
    golds = set([k for k, v in golds if v > 0])
    print "Gold: %s" % " ".join(list(golds))
    print "P@1AV: %.3f  %.3f  %.3f" % (brow['p@1av'], orow['p@1av'], row['p@1av'])
    print "P@3AV: %.3f  %.3f  %.3f" % (brow['p@3av'], orow['p@3av'], row['p@3av'])
    print "Table:"
    print "\\hline"
    print " & ".join("{    %-20s}" % c for c in ('OOC', '\\balAddCos', '\\ourmeas', '\\ourmeasparam')) + " \\\\"
    print "\\hline\\hline"
    print "\\multicolumn{4}{|c|}{%s}\\\\" % (row['sentence'].replace(word, "{\\bf %s}" % word))
    print "\\hline"
    for i in range(1, 6):
        cols = [t["sub0%d" % i][brow['ident']] for t in tables]
        print " & ".join("{%s%-20s}" % (bf(c in golds), c) for c in cols) + " \\\\"
    print "\\hline"
    print

