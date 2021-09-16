import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable

import time


class RNNPred(nn.Module):
	def __init__(self, x_dim, nlayers=1, hiddenUnits=48 ):
		super(RNNPred, self).__init__()
		self.nLayers = nlayers
		self.hiddenUnits = hiddenUnits
		#self.rnn = nn.RNN(x_dim, hiddenUnits, nlayers, bias=True)
		self.rnn = nn.LSTM(x_dim, hiddenUnits)
		self.fc = nn.Sequential(  nn.Linear(hiddenUnits, x_dim), nn.Sigmoid(), nn.Linear(x_dim, x_dim))

	def forward(self, x,  batchsize, y_len):
		#mask = torch.zeros((x.shape[0], x.shape[1] + y_length, y.shape[2]))#batchsize, x_len + y_len, output_dim

		#rnn
		#h = torch.zeros(self.nLayers, batchsize, self.hiddenUnits).double()
		h = ( torch.zeros(self.nLayers, batchsize, self.hiddenUnits).double(), torch.zeros(self.nLayers, batchsize, self.hiddenUnits).double())
		outputs, hn  = self.rnn(x,h)
		outputs = self.fc(outputs)
		#output = mask * output

		return outputs[-y_len:,:,:], hn

	### sample make sure that the forward part without backtraking is done
	def sample(self, x, y_len, h):

		predictions = []
		for i in range(y_len): # Because we take the first prediction from previous
			#print(x.shape) #[1, 701, 2]
			output, h = self.rnn(x, h)
			x = self.fc(output)
			predictions.append(x)
		return predictions