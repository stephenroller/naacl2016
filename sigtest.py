#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

model1 = "orensm"
model2 = "uniform_dotted"

#column = "p@3av"
#column = "p@1av"
column = "gap"

data = "semeval_all"


for data in ['semeval_all', 'coinco', 'twsi2']:
    out1 = pd.read_table("lemma_predictions/%s/%s.tsv" % (data, model1))
    out2 = pd.read_table("lemma_predictions/%s/%s.tsv" % (data, model2))
    for column in ['gap', 'p@1av', 'p@3av']:


        samples1 = np.array(out1[column])
        samples2 = np.array(out2[column])

        mask = ~(np.isnan(samples1) | np.isnan(samples2))
        samples1 = samples1[mask]
        samples2 = samples2[mask]

        statistic, pvalue = wilcoxon(samples1, samples2)

        print "Means: %.3f , %.3f" % (samples1.mean(), samples2.mean())

        print "Comparing %s vs %s on %s [%s]" % (model1, model2, data, column)
        if pvalue < 0.01:
            print "Significant, p = %.3f" % pvalue
        else:
            print "Not significant %.3f" % pvalue

        print
