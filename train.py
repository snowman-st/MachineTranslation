# -*- coding: utf-8 -*-

import logging
import argparse
import torch.nn as nn
import torch
import pickle
from tqdm import tqdm

from torch.autograd import Variable

from utils.batch_make import getBatches,gettestBatches
from models.Seq2Seq_Rnn import Encoder,Decoder,Seq2Seq_rnn

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_file',type=str,default= 'data/data/eng-fra.train')
	parser.add_argument('--dev_file',type=str,default= 'data/data/eng-fra.dev')
	parser.add_argument('--best_model_path',type=str,default='saved/models/')
	parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--epochs',type=int,default=20)
	parser.add_argument('--batch_size',type=int,default=64)
	parser.add_argument('--embedding_size',type=int,default=300)
	parser.add_argument('--hidden_size',type=int,default=128)
	parser.add_argument('--src_vocab_size',type=int,default=25000)
	parser.add_argument('--trg_vocab_size',type=int,default=40000)


	return parser.parse_args()

def train():
	opt = parse_args()
	print('begin reading......')
	# train_batch,dev_batch,PAD_INDEX = getBatches(opt.train_file,opt.dev_file,opt.batch_size,opt.device)
	train_batch,dev_batch,PAD_INDEX,s,t = gettestBatches(opt.batch_size,opt.device)
	opt.src_vocab_size = s
	opt.trg_vocab_size = t
	print('end reading......')

	encoder = Encoder(opt).to(opt.device)
	decoder = Decoder(opt).to(opt.device)
	model = Seq2Seq_rnn(encoder,decoder).to(opt.device)
	model.train()
	optimizer = torch.optim.Adam(model.parameters())
	criterion = nn.NLLLoss(ignore_index=PAD_INDEX)  #pad_token==1
	best_loss = 8. 

	for epoch in range(opt.epochs):
		for senten_pairs in tqdm(train_batch):
			src = senten_pairs.src
			trg = senten_pairs.trg
			optimizer.zero_grad()
			output = model(src,trg,opt)
			loss = criterion(output[1:].view(-1,opt.trg_vocab_size),trg[1:].contiguous().view(-1))  
			loss.backward()
			optimizer.step()
		devloss = eval(model,dev_batch,criterion,opt)
		if devloss < best_loss:
			best_loss = devloss
			torch.save(model,opt.best_model_path+'rnn.pkl')
			print('The {} epoch with train loss:{}//test loss:{}'.format(epoch+1,loss,devloss))
			print('model saved in {}!'.format(opt.best_model_path+'rnn.pkl'))
		print('The {} epoch of training with:'.format(epoch+1))
		print('train loss:{}//test loss:{}'.format(loss.item(),devloss.item()))

def eval(model,dev_batch,criterion,opt):
	# model = torch.load()
	devloss = 0.
	count = 0
	for sentpairs in dev_batch:
		devsrc = sentpairs.src
		devtrg = sentpairs.trg
		with torch.no_grad():
			testout = model(devsrc,devtrg,opt)
		count += 1
		devloss += criterion(testout[1:].view(-1,opt.trg_vocab_size),devtrg[1:].contiguous().view(-1))

	return devloss/count

if __name__ == '__main__':
	train()

