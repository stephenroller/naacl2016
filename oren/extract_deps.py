# extract dependency pairs for word2vecf from either a conll file or a tree file
# modified to stanford dependencies instead of google universal-treebank annotation scheme.
# zcat treebank.gz |python extract_deps.py |gzip - > deps.gz

import sys
import re
from collections import defaultdict

def read_sent(fh, format):
    if format == 'conll':
        return read_conll(fh)
    elif format == 'stanford':
        return read_stanford(fh)
    else:
        raise LookupError('unknown sentence format %' % format)

#conll format example:
#1       during  _       IN      IN      _       7       prep
def read_conll(fh):
   root = (0,'*root*',-1,'rroot')
   tokens = [root]
   for line in fh:
      if lower: line = line.lower()
      tok = line.strip().split()
      if not tok:
         if len(tokens)>1: yield tokens
         tokens = [root]
      else:
         tokens.append((int(tok[0]),tok[1],int(tok[6]),tok[7]))
   if len(tokens) > 1:
      yield tokens
      
line_extractor = re.compile('([a-z:]+)\(.+-(\d+), (.+)-(\d+)\)')      
# stanford parser output example:  
# num(Years-3, Five-1) 
def read_stanford(fh):
   root = (0,'*root*',-1,'rroot')
   tokens = [root]
   for line in fh:
      if lower: line = line.lower()
      tok = line_extractor.match(line)
      if not tok:
         if len(tokens)>1: yield tokens
         tokens = [root]
      else:
         tokens.append((int(tok.group(4)),tok.group(3),int(tok.group(2)),tok.group(1)))
   if len(tokens) > 1:
      yield tokens      

def read_vocab(fh):
   v = {}
   for line in fh:
      if lower: line = line.lower()
      line = line.strip().split()
      if len(line) != 2: continue
      if int(line[1]) >= THR:
         v[line[0]] = int(line[1])
   return v


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: parsed-file | %s <conll|stanford> <vocab-file> [<min-count>] > deps-file \n" % sys.argv[0])
        sys.exit(1)
        
    format = sys.argv[1]
    vocab_file = sys.argv[2]
    
    try:
        THR = int(sys.argv[3])
    except IndexError: THR=100

    lower=True
    
    vocab = set(read_vocab(file(vocab_file)).keys())
    print >> sys.stderr,"vocab:",len(vocab)
    for i,sent in enumerate(read_sent(sys.stdin, format)):
       if i % 100000 == 0: print >> sys.stderr,i
       for tok in sent[1:]:
          print "***"
          print sent
          print "---"
          par_ind = tok[2] 
          par = sent[par_ind]
          m = tok[1]
          if m not in vocab: continue
          rel = tok[3]

          if rel == 'prep' or rel == 'adpmod': continue # this is the prep. we'll get there (or the PP is crappy)
          if (rel == 'adpobj' or rel == 'pobj') and par[0] != 0:
    
             ppar = sent[par[2]]
             rel = "%s:%s" % (par[3],par[1])
             h = ppar[1]
          else:
             h = par[1]
          if h not in vocab and h != '*root*': continue
          if h != '*root*': print h,"_".join((rel,m))
          print m,"I_".join((rel,h))
    
    
    
    
