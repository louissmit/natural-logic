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