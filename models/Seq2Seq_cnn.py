#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self,*args,**kwargs):
		super(Encoder,self).__init(opt)
		self.embedding_size = opt.embedding_size
		self.vocab_size = opt.src_vocab_size
		self.seq_len = opt.max_seq_len
		self.hidden_size = opt.hidden_size
		self.filter_size = opt.filter_size
		self.device = opt.device

		self.word_embed = nn.Embedding(self.vocab_size,self.embedding_size)
		self.pos_embed  = nn.Embedding(self.seq_len,self.embedding_size)

		self.emb2hid = nn.Linear(self.embedding_size,self.hidden_size)
		self.hid2emb = nn.Linear(self.hidden_size,self.embedding_size)

		self.convs = nn.ModuleList([nn.Conv2d(
			in_channels = opt.in_channels,
			out_channels = self.output_channels,
			kernel_size = (self.filter_size,self.hidden_size),
			padding = ((self.filter_size - 1) // 2 , 0))
			for _in range(opt.n_conv_layers)])

		self.scale = torch.sqrt(torch.tensor([0.5])).to(self.device)
		self.dropout = nn.Dropout(opt.dropout)

	def forward(self,inputs,src_mask):
		'''
		  	inputs :  N * T
			src_mask: N * T
		'''
		position = torch.arange(0,inputs.shape[1]).unsqueeze(0).repeat(inputs.shape[0],1).to(self.device)
		position = position * src_mask

		word_embedding = self.word_embed(inputs)
		pos_embedding  = self.pos_embed(position)
		embedding = word_embedding + pos_embedding
		# N * T * E

		input_feature = self.emb2hid(embedding).unsqueeze(1)
		# N * T * E  ———> N * 1 * T * H

		for conv in self.convs:
			conv_out = self.dropout(conv(input_feature))
			conv_out = F.glu(conv_out)
			input_feature = (input_feature + conv_out) * self.scale

		conved = self.hid2emb(input_feature.squeeze())
		combined = (conved + embedding) * self.scale
		#  值得探索 为什么每次残差连接之后要进行一下scale

		return conved , combined

class Decoder(nn.Module):
	def __init__(self,*args,**kwargs):
		super(Decoder,self).__init__(opt)
		self.hidden_size = opt.hidden_size
		self.embedding_size = opt.embedding_size
		self.vocab_size = opt.trg_vocab_size
		self.device = opt.device
		self.filter_size = opt.filter_size

		self.word_embed = nn.Embedding(self.vocab_size,self.embedding_size)

		self.emb2hid = nn.Linear(self.embedding_size,self.hidden_size)
		self.hid2emb = nn.Linear(self.hidden_size,self.embedding_size)


		self.attnemb2hid = nn.Linear(self.embedding_size,self.hidden_size)
		self.attnhid2emb = nn.Linear(self.hidden_size,self.embedding_size)

		self.convs = nn.ModuleList([nn.Conv2d(
			in_channels = opt.in_channels,
			out_channels = opt.out_channels,
			kernel_size = opt.filter_size,
			)
			for _ in range(opt.n_convlayers)])

		self.scale = torch.sqrt(torch.tensor([0.5])).to(self.device)
		self.dropout = nn.Dropout(opt.dropout)
		self.out = nn.Linear(self.embedding_size,self.vocab_size)

	def attention(self,decoder_embedded,decoder_conved,encoder_conved,encoder_combined):
		'''
			decoder_embedded  : N * P * E   为了区别于src的长度，我们记trg的长度为  P
			decoder_conved : 	N * 1 * P * H
			encoder_conved :  	N * T * E
			encoder_combined :	N * T * E
		'''
		decoder_conved = self.attnhid2emb(decoder_conved.squeeze())
		decoder_combined = (decoder_conved + decoder_embedded) * self.scale
		#  N * P * E

		attn_weights = decoder_combined.bmm(encoder_conved.permute(0,2,1))
		# N * P * T
		attn_weights = F.softmax(attn_weights,dim=2)

		decoder_attn = torch.bmm(attn_weights,encoder_combined)
		# N * P * E

		decoder_attn = self.attnemb2hid(decoder_attn)
		attn_combined = (decoder_conved + decoder_attn)

		return attn_combined

	def forward(self,trg,encoder_conved,encoder_combined):
		'''
			trg : N * P
			encoder_conved :  	N * T * E
			encoder_combined :	N * T * E
		'''
		word_embedding = self.word_embedding(trg)

		input_feature = self.emb2hid(word_embedding)

		for conv in self.convs:
			'''
				需向前 pad 两个字符
			'''
			padding = torch.zeros(input_feature.shape[0],self.filter_size-1,input_feature.shape[2])
			padded_input_feature = torch.cat((padding,input_feature),dim=1)

			conv_out = self.dropout(conv(input_feature))
			conv_out = F.glu(conv_out)

			attn_combined = self.attention(word_embedding,conv_out,encoder_conved,encoder_combined)

			input_feature = (attn_combined + input_feature) * self.scale

		conved = self.hid2emb(input_feature)

		out = self.out(self.dropout(conved))
		return out

class Seq2Seq_CNN(nn.Module):
	def __init__(self,*args,**kwargs):
		super().__init__(encoder,decoder)
		self.encoder = encoder
		self.decoder = decoder

	def forward(self,src,trg,opt=None):
		src_mask = self.create_mask(src,1)
		encoder_conved,encoder_combined = self.encoder(src,src_mask)
		output = self.decoder(trg,encoder_conved,encoder_combined)
		return ouput

	def create_mask(self,src,pad_index):
		# src: sentence_len * batch_size 
		return (src != pad_index).permute(1,0)