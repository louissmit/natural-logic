from os import listdir
from os.path import isfile, join
from tensortree import Leaf, Tree

class DataSet():
    def __init__(self, fname):
        def make(l, w):
            if w in l:
                return l.index(w)
            else:
                l.append(w)
                return len(l)-1
                    
        self.vocab = []
        self.rels = []
        self.pairs = set()
        for l in list(open(fname, 'r')):
            if not l[0] is '%' and l.split():
                [rel, w1, w2] = l.split()                        
                self.pairs.add((
                    make(self.rels, rel), 
                    make(self.vocab, w1), 
                    make(self.vocab, w2))
                )
        self.sets = {}

    def load_sents(self, fpath):
        for f in listdir(fpath):
            fname = join(fpath,f)
            if isfile(fname) and f[-3:] == 'tsv':
                # add this file to data sets
                name = f[:-4]
                self.sets[name] = []
                for l in list(open(fname, 'r')):
                    [r, t1, t2] = [w.strip() for w in l.split('\t')]
                    self.sets[name].append(
                        (self.rels.index(r), self.parse(t1), self.parse(t2))
                    )

    def parse(self, treestr):
        """ Hacky parser """
        """ ( some hippo ) ( not bark ) """

        def p(inc, rest):
            """ PseudoProlog treeparser """
            if not rest:
                return inc, None
            if rest[0] == '(':
                i1, r = p([], rest[1:])
                i2, r2 = p([], r)
                return inc + [i1] + i2, r2
            elif rest[0] == ')':
                return inc, rest[1:]
            else:
                i, r = p([], rest[1:])
                return inc + [rest[0]] + i, r
        
        def walk(t):
            """ Build binary tree from nested list """
            if not type(t) is list:
                # lookup word
                return Leaf(self.vocab.index(t))
            elif len(t) == 2:
                return Tree(walk(t[0]), walk(t[1]))
            else:
                raise Exception('nonbinary tree')

        nest, _ = p([], treestr.split())
        return walk(nest)


if __name__ == "__main__":
    d = DataSet('wordpairs-v2.tsv')
    d.load_sents('data-4')
    print d.sets.keys()
    print d.sets[d.sets.keys()[0]][0]
