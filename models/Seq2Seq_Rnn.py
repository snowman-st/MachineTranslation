# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
	def __init__(self,opt):
		super(Encoder,self).__init__()
		self.embedding_size = opt.embedding_size
		self.hidden_size = opt.hidden_size
		self.vocab_size = opt.src_vocab_size
		self.embed = nn.Embedding(self.vocab_size,self.embedding_size)
		self.gru = nn.GRU(self.embedding_size,self.hidden_size,num_layers=1,bidirectional=True,batch_first = False)
		self.bi2uni = nn.Linear(self.hidden_size * 2, self.hidden_size)

	def forward(self,inputs):
		embedded = self.embed(inputs)
		# embedded  N * T * E
		outputs,hn = self.gru(embedded)
		# outputs   N * T * (E*2)
		# hn[0]  N * 1 * (E*2)
		outputs = self.bi2uni(outputs)
		return outputs,hn[0]


class Decoder(nn.Module):
	def __init__(self,opt):
		super(Decoder,self).__init__()
		self.hidden_size = opt.hidden_size
		self.embedding_size = opt.embedding_size
		self.vocab_size = opt.trg_vocab_size
		self.device = opt.device

		self.embed = nn.Embedding(self.vocab_size,self.embedding_size)
		self.gru = nn.GRU(self.embedding_size+self.hidden_size,self.hidden_size,batch_first=False)
		self.attn = nn.Linear(self.hidden_size,self.hidden_size)

		self.out = nn.Linear(self.hidden_size,self.vocab_size)

	def forward(self,inputs,pre_hidden,outputs):
		embedded = self.embed(inputs)
		# embedded N * 1 * E
		# print(pre_hidden.shape)
		# print(outputs.shape)
		# assert 1<0
		attn_weights = F.softmax(pre_hidden.permute(1,0,2).bmm(self.attn(outputs).permute(1,2,0)),dim=2)
		# attn_weights  N * 1 * T
		attn_combine = attn_weights.bmm(outputs.permute(1,0,2))
		# attn_combine N * 1 * H
		gru_input = torch.cat((embedded.unsqueeze(0),attn_combine.permute(1,0,2).contiguous()),2).contiguous()
		_,hn = self.gru(gru_input,pre_hidden.contiguous())
		out = self.out(hn[0])
		return F.log_softmax(out),hn[0]

	def initHidden(self):
		return Variable(torch.zeros(1,1,self.hidden_size)).to(self.device)

class Seq2Seq_rnn(nn.Module):
	def __init__(self,encoder,decoder):
		super(Seq2Seq_rnn,self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self,encoder_input,decoder_input,opt):
		return decode(self.encoder,self.decoder,encoder_input,decoder_input,opt)


def decode(encoder,decoder,encoder_input,decoder_input,opt):
	'''
	encoder_input:  T * N
	decoder_input:  T * N

	'''
	SOS_TOKEN = 2
	EOS_TOKEN = 3

	src_len = encoder_input.shape[0]
	trg_len = decoder_input.shape[0]
	batch_size = decoder_input.shape[1]

	encoder_outputs,hn = encoder(encoder_input)
	# encoder_outputs   N*T*(E*2)


	decoder_hidden = decoder.initHidden().expand(-1,batch_size,-1)
	decoder_inputs = Variable(decoder_input.data[0,:]).to(opt.device)   # SOS_TOKEN
	outputs = Variable(torch.zeros(trg_len,batch_size,opt.trg_vocab_size)).to(opt.device)

	for di in range(1,trg_len):
		decoder_output, decoder_hidden = decoder(
			decoder_inputs, decoder_hidden, encoder_outputs)
		outputs[di] = decoder_output
		decoder_inputs = decoder_output.data.max(1)[1].to(opt.device)
		decoder_hidden = decoder_hidden.unsqueeze(0)
	return outputs

	
