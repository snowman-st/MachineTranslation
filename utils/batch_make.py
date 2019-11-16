# -*- coding: utf-8 -*-

import torchtext.data as td
from torchtext import datasets
import spacy

def getBatches(trainfile,devfile,batch_size,device):

	UNK_TOKEN = '<UNK>'
	PAD_TOKEN = '<PAD>'
	SOS_TOKEN = '<S>'
	EOS_TOKEN = '</S>'

	spacy_fr = spacy.load('fr_core_news_sm')
	spacy_en = spacy.load('en_core_web_sm')
	def tokenize_fr(text):
		return [tok.text for tok in spacy_fr.tokenizer(text)]

	def tokenize_en(text):
		return [tok.text for tok in spacy_en.tokenizer(text)]

	SRC_TEXT = td.Field(fix_length=45,tokenize=tokenize_en,unk_token=UNK_TOKEN,pad_token=PAD_TOKEN,init_token=None,eos_token=EOS_TOKEN)
	TRG_TEXT = td.Field(fix_length=50,tokenize=tokenize_fr,unk_token=UNK_TOKEN,pad_token=PAD_TOKEN,init_token=SOS_TOKEN,eos_token=EOS_TOKEN)

	train = datasets.TranslationDataset(trainfile, exts=('.en','.fr'), fields=(SRC_TEXT,TRG_TEXT))
	dev   = datasets.TranslationDataset(devfile, exts=('.en','.fr'), fields=(SRC_TEXT,TRG_TEXT))
	#这里的path是两种语言的公共path，exts指的是两种语言的扩展名


	#待测试
	# alldata = datasets.TranslationDataset(allfile,exts=('.en','.fr'),fields=(SRC_TEXT,TRG_TEXT))
	# train,dev = alldata.split(split_ratio=0.7)

	SRC_TEXT.build_vocab(train.src,max_size=25000)
	TRG_TEXT.build_vocab(train.trg,max_size=40000)
	#  .src 和 .trg 是Translationdataset内部给加进去的两个固定属性

	# test_vocab = SRC_TEXT.vocab
	# print(test_vocab.stoi[EOS_TOKEN])  # 2
	# print('-------')
	# s_vocab = TRG_TEXT.vocab
	# print(s_vocab.stoi[EOS_TOKEN])   # 3
	# print(s_vocab.stoi[SOS_TOKEN])   # 2


	train_iter = td.BucketIterator(dataset = train, batch_size = batch_size,sort_key = lambda x: data.interleave_keys(len(x.src), len(x.trg)),device = device)
	dev_iter = td.BucketIterator(dataset = dev, batch_size = 1 , train=False,device = device)
	#train_iter.src.shape = [45 *32]
	#train_iter.trg.shape = [50 *32]


	'''
	interleave_keys
	交错排序为排序键中的每个列或列的子集赋予相同的权重。如果多个查询使用不同的列作为筛选条件，
	则通常可以使用交错排序方式来提高这些查询的性能。譬如当查询对辅助排序列使用限制性谓词时，与复合排序相比，交错排序可显著提高查询的性能。

	'''
	return train_iter,dev_iter



# from torchtext import data, datasets

# if True:
#	 import spacy
#	 spacy_de = spacy.load('de')
#	 spacy_en = spacy.load('en')

#	 def tokenize_de(text):
#		 return [tok.text for tok in spacy_de.tokenizer(text)]

#	 def tokenize_en(text):
#		 return [tok.text for tok in spacy_en.tokenizer(text)]

#	 BOS_WORD = '<s>'
#	 EOS_WORD = '</s>'
#	 BLANK_WORD = "<blank>"
#	 SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
#	 TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)
#	 MAX_LEN = 100
#	 train, val, test = datasets.IWSLT.splits(
#		 exts=('.de', '.en'), fields=(SRC, TGT), 
#		 filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
#			 len(vars(x)['trg']) <= MAX_LEN)
#	 MIN_FREQ = 2
#	 SRC.build_vocab(train.src, min_freq=MIN_FREQ)
#	 TGT.build_vocab(train.trg, min_freq=MIN_FREQ)