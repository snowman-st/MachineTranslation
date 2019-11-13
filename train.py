# -*- coding: utf-8 -*-

import logging
import argparse


def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_file',type=str,default= 'data/data/eng-fra.train')
	parser.add_argument('--dev_file',type=str,default= 'data/data/eng-fra.dev')
	parser.add_argument('--epochs',type=int,default=20)
	parser.add_argument('--batch_size',type=int,default=32)
	parser.add_argument('--embedding_size',type=int,default=300)
	parser.add_argument('--hidden_size',type=int,default=128)
	
