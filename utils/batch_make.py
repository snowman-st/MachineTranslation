# -*- coding: utf-8 -*-

import torchtext.data as td
from torchtext import datasets

def getBatches(trainfile,devfile,device)

	SRC_TEXT = td.Field(fix_length=45,tokenize=lambda x : x.split())
	TRG_TEXT = td.Field(fix_length=50,tokenize=lambda x : x.split())

	train = datasets.TranslationDataset(trainfile, exts=('.en','.fr'), fields=(SRC_TEXT,TRG_TEXT))
	dev   = datasets.TranslationDataset(devfile, exts=('.en','.fr'), fields=(SRC_TEXT,TRG_TEXT))

	SRC_TEXT.build_vocab(train,max_size=25000)
	TRG_TEXT.build_vocab(train,max_size=40000)

	train_iter = td.BucketIterator(dataset = train, batch_size = 32,sort_key = lambda x: data.interleave_keys(len(x.src), len(x.trg)),device = device)
	dev_iter = td.BucketIterator(dataset = dev, batch_size = 64 , train=False,device = device)

	'''
	interleave_keys
	交错排序为排序键中的每个列或列的子集赋予相同的权重。如果多个查询使用不同的列作为筛选条件，
	则通常可以使用交错排序方式来提高这些查询的性能。譬如当查询对辅助排序列使用限制性谓词时，与复合排序相比，交错排序可显著提高查询的性能。

	'''
	return train_iter,dev_iter