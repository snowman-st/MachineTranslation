#-*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Seq2Seq_trans(nn.Module):
	def __init__(self,opt,*args,**kwargs):
		super().__init__()
		self.embedding_size = opt.d_model
		self.vocab_size = opt.trg_vocab_size
		self.src_embed = nn.Embedding(opt.src_vocab_size,self.embedding_size)
		self.trg_embed = nn.Embedding(opt.trg_vocab_size,self.embedding_size)
		self.transformer = nn.Transformer(self.embedding_size, opt.n_heads,opt.num_encoder_layers,opt.num_decoder_layers)
		self.out = nn.Linear(self.embedding_size,self.vocab_size)
	def forward(self,src,trg,opt=None):
		src = self.src_embed(src).permute(1,0,2)
		trg = self.trg_embed(trg).permute(1,0,2)
		out = self.transformer(src,trg).permute(1,0,2)
		output = self.out(out)
		return output