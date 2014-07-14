from os import listdir
from os.path import isfile, join
from tensortree import Leaf, Tree
from random import choice

class DataSet():
    def __init__(self, fname, relations_file=None):
        def make(l, w):
            if w in l:
                return l.index(w)
            else:
                l.append(w)
                return len(l)-1
                    
        self.vocab = []
        self.rels = []
        self.pairs = set()

        if not relations_file:
            # Bowman style
            for l in list(open(fname, 'r')):
                if not l[0] is '%' and l.split():
                    [rel, w1, w2] = l.split()                        
                    self.pairs.add((
                        make(self.rels, rel), 
                        make(self.vocab, w1), 
                        make(self.vocab, w2))
                    )
        else:
            # Seperate vocab file style
            for l in list(open(fname, 'r')):
                if not l[0] is '%':
                    w = l.split()
                    make(self.vocab, w[0])
            for l in list(open(relations_file, 'r')):
                if not l[0] is '%':
                    r = l.split()
                    make(self.rels, r[0])

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
                        (self.parse(t1), self.parse(t2), self.rels.index(r))
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
            elif len(t) == 1:
                return walk(t[0])
            else:
                raise Exception('nonbinary tree: '+str(len(t))+'\n'+str(t))

        nest, _ = p([], treestr.split())
        return walk(nest)

    def sample_balanced(self, dataset, nr):
        """ Sample (with replacement) `nr` items from every class in `dataset`"""
        classes = {}
        for i in set(point[2] for point in self.sets[dataset]):
            classes[i] = [point for point in self.sets[dataset] if point[2]==i]

        for _ in range(nr):
            for _, l in classes.iteritems():
                yield choice(l)




if __name__ == "__main__":
    d = DataSet('wordpairs-v2.tsv')
    d.load_sents('data-4')
    print d.sets.keys()
    print d.sets[d.sets.keys()[0]][0]


    d2 = DataSet('data-sick/vocab', relations_file='data-sick/vocab-relations')
    d2.load_sents('data-sick')
    print d2.sets.keys()
    print d2.sets[d2.sets.keys()[0]][0]
