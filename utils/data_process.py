# -*_ coding: utf-8 -*_

import pandas as pd
import codecs


In_file = '../data/data/eng-fra.txt'

with codecs.open(In_file, 'r') as f:
	lines = f.readlines()
	lens = len(lines)
	train_size = int(0.7 * lens)
	en_train = codecs.open('../data/data/eng-fra.train.en', 'a+',encoding='utf-8')
	fr_train = codecs.open('../data/data/eng-fra.train.fr', 'a+',encoding='utf-8')
	for i in range(train_size):
		en , fr = lines[i].split('\t')
		en_train.write(en+'\n')
		fr_train.write(fr)
		
	en_dev = codecs.open('../data/data/eng-fra.dev.en', 'a+',encoding='utf-8')
	fr_dev = codecs.open('../data/data/eng-fra.dev.fr', 'a+',encoding='utf-8')
	for i in range(train_size,lens):
		en,fr = lines[i].split('\t')
		en_dev.write(en+'\n')
		fr_dev.write(fr)

	en_train.close()
	fr_train.close()
	en_dev.close()
	fr_dev.close()
	
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