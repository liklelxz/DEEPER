import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from attn import *

class RNN(nn.Module):
	def __init__(self, input_size=1, output_size=1, hidden_layer_size=200, type_actifun='relu'):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size
		self.output_size = output_size
		self.rnn = nn.RNN(input_size, hidden_layer_size)

		if type_actifun=='relu':
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.ReLU())
		else:#Sigmoid
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.Sigmoid())

	def forward(self, input_seq, batch_size,hidden_cell, pred_length):
		#only h, not c
		hidden_cell = hidden_cell[0]
		lstm_out, hidden_cell = self.rnn(input_seq.view(len(input_seq) ,batch_size, -1), hidden_cell)

		predictions = self.linear(lstm_out.view(-1, self.hidden_layer_size))

		predictions = predictions.reshape(len(input_seq),batch_size, self.output_size)
		return predictions[-pred_length:]


class LSTM(nn.Module):
	def __init__(self, input_size=1, output_size=1, hidden_layer_size=200, type_actifun='relu'):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size
		self.input_size = input_size
		self.output_size = output_size
		self.lstmEnc = nn.LSTM(input_size, hidden_layer_size)

		self.rnnDec = nn.RNN(output_size, hidden_layer_size)

		if type_actifun=='relu':
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.ReLU())
		else:#Sigmoid
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.Sigmoid())

	def forward(self, input_seq, output_seq, batch_size,hidden_cell, pred_length):

		#Encoder Forward
		lstm_outEnc, hidden_cellEnc = self.lstmEnc(input_seq.view(len(input_seq) ,batch_size, self.input_size ), hidden_cell)

		#Decoder Forward
		predictions = []


		# If we have more than one feature in the input sequence and only one feature (the predicting variable) in the output sequence
		# then we take only the first feature
		if input_seq.shape[-1] != output_seq.shape[-1]:
			#print("dec inp ", input_seq.shape)
			dec_input = input_seq[-1,:,0] #last step, all batches and first feature
		else:
			dec_input = input_seq[-1]# + output_seq[:-1]
		
		#print(dec_input.shape, input_seq.shape)
		
		hidden_cellDec = hidden_cellEnc[0]#h,c
		for step in range(pred_length):

			#print(dec_input.shape)
			lstm_outDec, hidden_cellDec = self.rnnDec(dec_input.view(1,batch_size, -1), hidden_cellDec)
			dec_input = self.linear(lstm_outDec)
			predictions.append(dec_input)
		#predictions = self.linear(lstm_out.view(-1, self.hidden_layer_size))

		#predictions = predictions.reshape(len(input_seq),batch_size, self.output_size)
		
		#print(torch.cat(predictions[-pred_length:]).shape)
		return torch.cat(predictions[-pred_length:])



class RNNAttn(nn.Module):
	def __init__(self, input_size=1, output_size=1, hidden_layer_size=200, type_actifun='relu'):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size
		self.output_size = output_size
		self.rnn = nn.RNN(input_size, hidden_layer_size)

		if type_actifun=='relu':
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.ReLU())
		else:#Sigmoid
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.Sigmoid())

	def forward(self, input_seq, batch_size,hidden_cell, pred_length):
		#only h, not c
		hidden_cell = hidden_cell[0]
		lstm_out, hidden_cell = self.rnn(input_seq.view(len(input_seq) ,batch_size, -1), hidden_cell)

		predictions = self.linear(lstm_out.view(-1, self.hidden_layer_size))

		predictions = predictions.reshape(len(input_seq),batch_size, self.output_size)
		return predictions[-pred_length:]


class LSTMAttn(nn.Module):
	def __init__(self, input_size=1, output_size=1, hidden_layer_size=200, type_actifun='relu'):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size
		self.output_size = output_size

		self.lstmEnc = nn.LSTM(input_size, hidden_layer_size)

		self.rnnDec = nn.RNN(output_size, hidden_layer_size)


		self.attnStep = Attention(hidden_layer_size)

		
		#Due to attention, this layer is applied in the AATN module
		if type_actifun=='relu':
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.ReLU())
		else:#Sigmoid
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.Sigmoid())

		self.flgPrint= 1
		

	def forward(self, input_seq, output_seq, batch_size,hidden_cell, pred_length):

		#Encoder Forward
		lstm_outEnc, hidden_cellEnc = self.lstmEnc(input_seq[:-1].view(len(input_seq[:-1]) ,batch_size, -1), hidden_cell)

		#Decoder Forward
		predictions = []


		dec_input = input_seq[-1]# + output_seq[:-1]
		#dec_input = output_seq[0]
		
		#print(dec_input.shape, input_seq.shape)
		
		hidden_cellDec = hidden_cellEnc[0]#h,c
		for step in range(pred_length):
			
			
			dec_input, hidden_cellDec = self.rnnDec(dec_input.view(1,batch_size, -1), hidden_cellDec)
			
			dec_input, attnWeights = self.attnStep(lstm_outEnc.transpose(1,0), hidden_cellDec.transpose(1,0))#

			#print(dec_input.shape)
			
			dec_input = self.linear(dec_input)

			predictions.append(dec_input)
			#print(dec_input.view(1,batch_size, -1).shape, hidden_cellDec.shape)

		finalOutput = torch.stack(predictions,)#-pred_length
		if self.flgPrint==1:
			print(torch.cat(predictions).shape, torch.stack(predictions).shape)
			self.flgPrint=0
		return finalOutput