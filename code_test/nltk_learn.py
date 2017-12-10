# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: nltk_learn.py
@time: 2017/10/23 19:30
"""
import nltk
# nltk.download()
from nltk.corpus import wordnet as wn
panda = wn.synset('panda.n.01')
hyper = lambda s: s.hypernyms()
data = list(panda.closure(hyper))
print("data: ", data)