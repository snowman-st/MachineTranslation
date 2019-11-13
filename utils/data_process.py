# -*_ coding: utf-8 -*_

import pandas as pd
import codecs


In_file = '../data/data/eng-fra.txt'

with codecs.open(In_file, 'r') as f:
	lines = f.readlines()
	lens = len(lines)
	train_size = int(0.7 * lens)
	with codecs.open('../data/data/eng-fra.train','w+',encoding='utf-8') as t:
		for i in range(train_size):
			t.write(lines[i])

	with codecs.open('../data/data/eng-fra.dev','w+',encoding='utf-8') as d:
		for j in range(train_size,lens):
			d.write(lines[j])

df = pd.read_csv(In_file,sep='\t',names=['en','fr'])
en_corpus = df['en'].tolist()
fr_corpus = df['fr'].tolist()
print(max([len(s.split()) for s in en_corpus]))  #47
print(max([len(s.split()) for s in fr_corpus]))  #54


words = [s.split() for s in en_corpus]
wordset = set()
for w in words:
	wordset |= set(w)

print(len(wordset))     #english words: 25028

words = [s.split() for s in fr_corpus]
wordset = set()
for w in words:
	wordset |= set(w)

print(len(wordset))     #franch words: 41024