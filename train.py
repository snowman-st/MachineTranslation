# -*- coding: utf-8 -*-

import logging
import argparse
import torch.nn as nn
import torch
import pickle

from torch.autograd import Variable

from utils.batch_make import getBatches
from models.Seq2Seq_Rnn import Encoder,Decoder,Seq2Seq_rnn

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_file',type=str,default= 'data/data/eng-fra.train')
	parser.add_argument('--dev_file',type=str,default= 'data/data/eng-fra.dev')
	parser.add_argument('--best_model_path',type=str,default='saved/models/')
	parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--epochs',type=int,default=20)
	parser.add_argument('--batch_size',type=int,default=32)
	parser.add_argument('--embedding_size',type=int,default=300)
	parser.add_argument('--hidden_size',type=int,default=128)
	parser.add_argument('--src_vocab_size',type=int,default=25000)
	parser.add_argument('--trg_vocab_size',type=int,default=40000)


	return parser.parse_args()

def train():
	opt = parse_args()
	print('begin reading......')
	train_batch,dev_batch = getBatches(opt.train_file,opt.dev_file,opt.batch_size,opt.device)
	print('end reading......')

	encoder = Encoder(opt).to(opt.device)
	decoder = Decoder(opt).to(opt.device)
	model = Seq2Seq_rnn(encoder,decoder).to(opt.device)
	model.train()
	optimizer = torch.optim.Adam(model.parameters())
	criterion = nn.NLLLoss(ignore_index=1)
	best_loss = 8.
	for epoch in range(opt.epochs):
		for senten_pairs in train_batch:
			src = senten_pairs.src
			trg = senten_pairs.trg
			optimizer.zero_grad()
			output = model(src,trg,opt)
			loss = criterion(output[1:].view(-1,opt.trg_vocab_size),trg[1:].contiguous().view(-1))  #pad_token==1
			if loss < best_loss:
				best_loss = loss
				torch.save(model,opt.best_model_path+'rnn.pkl')

			loss.backward()
			optimizer.step()

def eval():
	# model = torch.load()
	pass

if __name__ == '__main__':
	train()

