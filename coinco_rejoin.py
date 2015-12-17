#!/usr/bin/env python

import sys
import argparse
import numpy as np
import pandas as pd

from itertools import chain

COINCO_FILE = "data/coinco/orig.xml"

def dict_merge(a, b):
    return dict(chain(a.iteritems(), b.iteritems()))

def parse_sentline(line):
    items = [x for x in line.split() if '=' in x]
    parsed = (x.split("=") for x in items)
    parsed = ((a, b.strip('"')) for a, b in parsed)
    return dict(parsed)

def parse_tokenline(line):
    tentative = parse_sentline(line)
    try:
        tentative['id'] = int(tentative['id'])
    except:
        return None
    return tentative

def get_genre(mascfile):
    known_genres = {
        'wsj_': 'non',
        'nyt': 'non',
        'lw1': 'fic',
        'bylichka': 'fic',
        'captured_moments': 'fic',
        'enron': 'email',
        'a1.': 'non',
        'apw': 'non',
    }

    for pat, cat in known_genres.iteritems():
        if pat.lower() in mascfile.lower(): return cat

    return "unk"


def main():
    parser = argparse.ArgumentParser('description')
    parser.add_argument('modelout', help='Output from lexsub.py')
    args = parser.parse_args()

    table = pd.read_table(args.modelout)

    tojoin = []

    with open(COINCO_FILE) as co:
        sentence_info = None
        for line in co:
            line = line.strip()
            if line.startswith('<sent '):
                sentence_info = parse_sentline(line)
            elif line.startswith('<token '):
                token_info = parse_tokenline(line)
                if token_info:
                    tojoin.append(dict_merge(token_info, sentence_info))

    joined = table.merge(pd.DataFrame(tojoin), on='id')
    joined['genre'] = joined['MASCfile'].apply(get_genre)
    import ipdb; ipdb.set_trace()
    joined.to_csv(sys.stdout, sep="\t", index=False)



if __name__ == '__main__':
    main()
