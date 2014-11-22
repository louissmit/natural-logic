__author__ = 'louissmit'

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from Tkinter import *
import Image, ImageDraw, ImageFont

# import matplotlib.colors as cm
from tsne import bh_sne
from dataset import DataSet

(S, T2, M2, b2, T, M, b, W) = pickle.load(open('params', 'r'))


X = W
print X.shape
# im = plt.imshow(X, interpolation='bilinear', cmap=cm.RdYlGn,
#                 origin='lower', extent=[-3,3,-3,3],
#                 vmax=abs(X).max(), vmin=-abs(X).min())
# im = plt.imshow(X, interpolation='none')
# plt.figure()
# im2 = plt.imshow(M2, interpolation='none')
# plt.colorbar(im, orientation='horizontal')
# plt.show()

res = bh_sne(X, perplexity=1.4)
print res
res[:,0] = [x - np.min(res[:,0]) for x in res[:,0]]
res[:,1] = [x - np.min(res[:,1]) for x in res[:,1]]

vocab = DataSet('wordpairs-v2.tsv').vocab

# master = Tk()
# w = Canvas(master, width=np.max(res[:,0])+100, height=np.max(res[:,1])+100)
# w.pack()

width = np.max(res[:,0]) + 80
height = np.max(res[:,1]) + 80
image1 = Image.new("RGB", (int(width), int(height)), 'white')
draw = ImageDraw.Draw(image1)
font= ImageFont.truetype('/System/Library/Fonts/Avenir.ttc',12)
for i, [x, y] in enumerate(res):
    draw.text((int(x)+10 , int(y)+ 10),vocab[i],fill='black', font=font)
    # w.create_text(x+50, y+50,
    #             text=vocab[i]
image1.save('test.png')
# mainloop()
