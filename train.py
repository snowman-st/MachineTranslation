# -*- coding: utf-8 -*-

import logging
import argparse
import torch.nn as nn
import torch

from torch.autograd import Variable

from utils.batch_make import getBatches
from models.Seq2Seq_Rnn import Encoder,Decoder,decode

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_file',type=str,default= 'data/data/eng-fra.train')
	parser.add_argument('--dev_file',type=str,default= 'data/data/eng-fra.dev')
	parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--epochs',type=int,default=20)
	parser.add_argument('--batch_size',type=int,default=32)
	parser.add_argument('--embedding_size',type=int,default=300)
	parser.add_argument('--hidden_size',type=int,default=128)
	parser.add_argument('--src_vocab_size',type=int,default=25000)
	parser.add_argument('--trg_vocab_size',type=int,default=40000)


	return parser.parse_args()

def main():
	opt = parse_args()
	print('begin reading......')
	train_batch,dev_batch = getBatches(opt.train_file,opt.dev_file,opt.batch_size,opt.device)
	print('end reading......')

	encoder = Encoder(opt).to(opt.device)
	decoder = Decoder(opt).to(opt.device)
	encoder_optimizer = torch.optim.Adam(encoder.parameters())
	decoder_optimizer = torch.optim.Adam(decoder.parameters())
	criterion = nn.NLLLoss()
	for tb in train_batch:
		src = tb.src
		trg = tb.trg
		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()

		loss = decode(encoder,decoder,src,trg,criterion,opt)
		print(loss)

		loss.backward()
		encoder_optimizer.step()
		decoder_optimizer.step()

	# print(train_batch)
if __name__ == '__main__':
	main()

