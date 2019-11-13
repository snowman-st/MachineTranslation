# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self,embedding_size,hidden_size,vocab_size):
		super(Encoder,self).__init__()
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size

		self.embed = nn.Embedding(vocab_size,self.embedding_size)
		self.gru = nn.GRU(self.embedding_size,self.hidden_size,num_layers=1,bidirectional=True,batch_first = True)

	def forward(self,inputs):
		embedded = self.embed(inputs)
		# embedded  N * T * E
		outputs,hn = self.gru(embedded)
		# outputs   N * T * (E*2)
		# hn[0]  N * 1 * (E*2)
		return outputs,hn[0]


class Decoder(nn.Module):
	def __init__(self,embedding_size,hidden_size,vocab_size,):
		super(Decoder,self).__init__()
		self.hidden_size = hidden_size

		self.embed = nn.Embedding(vocab_size,embedding_size)
		self.gru = nn.GRU(embedding_size+hidden_size,hidden_size,batch_first=True)
		self.attn = nn.Linear(hidden_size*2,hidden_size)

		self.out = nn.Linear(hidden_size,vocab_size)

	def forward(self,inputs,pre_hidden,outputs):
		embedded = self.embed(inputs)
		# embedded N * 1 * E
		attn_weights = F.softmax(pre_hidden.bmm(self.attn(outputs).permute(0,2,1)))
		# attn_weights  N * 1 * T
		attn_combine = attn_weights.bmm(outputs)
		# attn_combine N * 1 * H
		gru_input = torch.cat((embedded,attn_combine),2)
		_,hn = self.gru(gru_input,pre_hidden)
		out = self.out(hn[0])
		return out,hn[0]

	def initHidden(self):
		return Variable(torch.zeors(1,1,self.hidden_size))