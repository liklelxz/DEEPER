import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import datetime
import csv
from preprocessing.calculateError import *
from fileprocessor import *



#torch.manual_seed(0)




#Preprocessng
from sklearn.preprocessing import MinMaxScaler


parser = argparse.ArgumentParser(description='The Embedded Topic Model')

parser.add_argument('--data_path', type=str, default='./Data/With_Features/fire.csv', help='Path where the files are located')
parser.add_argument('--save_path', type=str, default='./Results/', help='path to save results')
parser.add_argument('--percentage', type=float, default=0.75, metavar='TP', help='training percentage. Rest is splitted in validation and testing sets')
parser.add_argument('--x-length', type=int, default=15, metavar='XL', help='previous time steps (default: 15)')
parser.add_argument('--y-length', type=int, default=5, metavar='YL', help='Time steps to predict (default: 5)')
parser.add_argument('--mini_batch', type=int, default=55, metavar='MB', help='Size of the mini_batch')
parser.add_argument('--epochs', type=int, default=1000, metavar='XL', help='previous time steps (default: 20)')
parser.add_argument('--input_dim', type=int, default=1, metavar='ID',help='steps to predict (default: 10)')
parser.add_argument('--output_dim', type=int, default=1, metavar='OD', help='steps to predict (default: 10)')
parser.add_argument('--hidden_layer', type=int, default=100, metavar='HL',help='number of hidden layers (default: 20)')
parser.add_argument('--type_training', type=str, default='AD', metavar='TT',help='Random mini batches (RB), all data (AD), fixed mini batches (FB)')
parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='LR', help='learning rate')
parser.add_argument('--type_rnn', type=str, default='lstm', metavar='TT',help='Random mini batches (RB), all data (AD), fixed mini batches (FB)')
parser.add_argument('--type_actifun', type=str, default='relu', metavar='AF',help='Select the activation function to the very last layer')

parser.add_argument('--bestIter', type=str, default='', metavar='BI',help='Iteration with std greater than quantile and smallest error')
parser.add_argument('--type_group', type=str, default='law', metavar='TG',help='TypeGroup')


clip = 0.01

args = parser.parse_args()

#Setting parameters
bestIter = args.bestIter
percentage = args.percentage
file_name = args.data_path
x_length = args.x_length
y_length = args.y_length
x_dim = args.input_dim
y_dim = args.output_dim
mini_batch_size = args.mini_batch
epochs = args.epochs
type_training = args.type_training
learning_rate = args.learning_rate
incident_group = args.type_group





#Create windows of sequences
def create_inout_sequences(input_data, xw, yw):
	'''
	Input: Sequence or dataFrame of one column
	Output: lists of X and list of Y
	'''
	in_seq = []
	out_seq = []
	L = len(input_data)
	for i in range(L-xw-yw):
		train_seq = input_data[i:i+xw]
		train_label = input_data[i+xw:i+xw+yw]
		in_seq.append(train_seq )
		out_seq.append(train_label)
	return in_seq, out_seq



def splitData(all_data, scaler):

	train_data_size = int(percentage*len(all_data))#431
	validation_data_size = train_data_size + int(0.5*(1-percentage)*len(all_data))
	train_data = all_data[:train_data_size]
	valid_data = all_data[train_data_size : validation_data_size]
	test_data = all_data[validation_data_size:]
	#print(len(train_data),len(valid_data), len(test_data))


	#Preprocessing
	
	scaler.fit(train_data.reshape(-1, 1))
	train_normalized = scaler.transform(train_data.reshape(-1, 1))
	valid_normalized = scaler.transform(valid_data.reshape(-1,1))
	test_normalized = scaler.transform(test_data.reshape(-1, 1))


	#Creating containers or pytorch variables
	train_data_normalized = torch.FloatTensor(train_normalized).view(-1)
	valid_data_normalized = torch.FloatTensor(valid_normalized).view(-1)
	test_data_normalized = torch.FloatTensor(test_normalized).view(-1)


	train_in, train_out = create_inout_sequences(train_data_normalized, x_length, y_length)
	val_in, val_out = create_inout_sequences(valid_data_normalized, x_length, y_length)
	test_in, test_out = create_inout_sequences(test_data_normalized, x_length, y_length)

	sizeTrain = len(train_in) 
	sizeTest = len(test_in)
	if type_training=='AD':
		mini_batch_size = sizeTrain
	num_batches = int(sizeTrain/mini_batch_size) 


	allIdx = range(sizeTrain)
	#Transform lists of x-y in arrays and move length to first dimension: torch.Size([len, batch_size, 1])
	seq_inOr_tra  = torch.stack(train_in).transpose(0,1)[:,:,None]
	seq_outOr_tra = torch.stack(train_out).transpose(0,1)[:,:,None]

	val_size = len(val_in)
	#print(val_size, np.quantile(train_normalized,0.9), np.quantile(valid_normalized,0.9), np.quantile(test_normalized,0.9))
	seq_inOr_val = torch.stack(val_in).transpose(0,1)[:,:,None]#torch.Size([15, 1334, 1])
	seq_outOr_val = torch.stack(val_out).transpose(0,1)[:,:,None]

	test_size = len(test_in)
	seq_inOr_tst = torch.stack(test_in).transpose(0,1)[:,:,None]#torch.Size([15, 1334, 1])
	seq_outOr_tst = torch.stack(test_out).transpose(0,1)[:,:,None]

	return seq_inOr_tst, seq_outOr_tst, np.quantile(valid_normalized,0.5), test_size, np.quantile(test_normalized,0.9)




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
		self.output_size = output_size
		self.lstm = nn.LSTM(input_size, hidden_layer_size)

		self.rnnDec = nn.RNN(input_size, hidden_layer_size)



		if type_actifun=='relu':
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.ReLU())
		else:#Sigmoid
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.Sigmoid())

	def forward(self, input_seq, batch_size,hidden_cell, pred_length):
		lstm_out, hidden_cell = self.lstm(input_seq.view(len(input_seq) ,batch_size, -1), hidden_cell)

		predictions = self.linear(lstm_out.view(-1, self.hidden_layer_size))

		predictions = predictions.reshape(len(input_seq),batch_size, self.output_size)
		return predictions[-pred_length:], hidden_cell

	def sample(self, dec_input,hidden_cellEnc, pred_length, batch_size):
		predictions = []
		
		hidden_cellDec = hidden_cellEnc[0]#h,c
		for step in range(pred_length):
			lstm_outDec, hidden_cellDec = self.rnnDec(dec_input.view(1,batch_size, -1), hidden_cellDec)
			dec_input = self.linear(lstm_outDec)
			predictions.append(dec_input)

		return torch.cat(predictions[-pred_length:])


def testing(seq_inOr, seq_outOr, size):
	model.eval()
	list_preds = []
	loss_test_batch = 0

	#TEST BATCH GUIDED

	####### REVIEW IF IT BATCH AND INDIVIDUAL TESTING GIVE SAME RESULTS, THEY DID
	test_preds = []
	with torch.no_grad():


		hidden = (torch.zeros(1, size, model.hidden_layer_size), torch.zeros(1, size, model.hidden_layer_size))
		preds_batch, hiddenEnc = model(seq_inOr,size,hidden, y_length)
		preds_batch = model.sample(seq_inOr[-1], hiddenEnc, y_length, size)
		single_loss = loss_function(preds_batch, seq_outOr)

		#####   single_loss = single_loss + torch.sum(seq_outOr)

		loss_test_batch += single_loss

	#print(loss_test_batch)
	
	# if RMSE boxplot, make inverse scale transform first
	lossBoxPlot = ((preds_batch-seq_outOr) **2 ).transpose(1,0).numpy().squeeze() # sequeeze to get rid of 3rd dimension  = 1



	loss_test = 0
	#INDIVIDUAL GUIDED
	for i in range(size):

		test_preds = []
		with torch.no_grad():


			hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

			preds, hiddenEnc =model(seq_inOr[:,i,:],1,hidden, y_length)
			preds = model.sample(seq_inOr[-1,i,:], hiddenEnc, y_length, 1)

			#preds=model(seq_in,1,hiddien, y_length)
			single_loss = loss_function(preds, seq_outOr[:,i,:,None])
			
			#####    single_loss = single_loss + torch.sum(seq_outOr)


			loss_test += single_loss

		#Adding as many lists as steps
		for step in range(y_length):
			test_preds.append(preds[step].detach().numpy().reshape((y_dim)))

		list_preds.append(test_preds)

	predsMetric = np.transpose(np.asarray(list_preds),[1,0,2])


	#### CALCULATING MAE, MSE TO SAVE THEM TO FILE
	pred_test_save = scaler.inverse_transform(predsMetric.squeeze().reshape(-1,1))#to flat the data
	pred_test_save= pred_test_save.reshape(y_length,-1).transpose(1,0)#to recover shape length,size -> size, lenght
	real_test_save = scaler.inverse_transform(seq_outOr.squeeze().detach().numpy().reshape(-1,1))
	real_test_save = real_test_save.reshape(y_length,-1).transpose(1,0)

	#### FIRST REVERS SCALE TRANSFORMATION
	err = RMSE(real_test_save,pred_test_save)
	maerr = MAE(real_test_save,pred_test_save)



	#print(preds[step].detach().numpy().squeeze().shape, np.asarray(list_preds).shape)
	return loss_test/size,predsMetric , loss_test_batch.item(), lossBoxPlot, np.mean(err)



def NUMPEAK(real_test_save, pred_test_save, qTest90):

	flgPeaksReal = np.where(real_test_save>=qTest90)
	numPeaksReal = np.sum(flgPeaksReal,axis=1)

	flgPeaksPred = np.where(pred_test_save>=qTest90)
	numPeaksPred = np.sum(flgPeaksPred,axis=1)

	ratioPeaks = np.mean(numPeaksPred/numPeaksReal)
	return ratioPeaks



def testModel(seq_inOr_tst, seq_outOr_tst, test_size, qTest90, save_path):
	loss_tst, pred_test, loss_tst_batch , loss_box_tst, err_avg= testing(seq_inOr_tst, seq_outOr_tst, test_size)
	diffMaxMin = pred_test.max() - pred_test.min()
	stdPreds = np.std(pred_test)
	loss_tst = loss_tst.item()#to bring from graph network space


	step_plot = 0
	plt.plot(np.concatenate((pred_test[step_plot],seq_outOr_tst[step_plot]),axis =1))
	plt.savefig("{}/bestV_PredvsReal_Test".format(save_path), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
	plt.clf()

	#Plot boxplot of losses
	dfBox = pd.DataFrame(loss_box_tst)
	dfBox.boxplot()
	plt.savefig(os.path.join(save_path,'bestV_boxplot'))
	plt.clf()

	#### CALCULATING MAE, MSE TO SAVE THEM TO FILE
	pred_test_save = scaler.inverse_transform(pred_test.squeeze().reshape(-1,1))#to flat the data
	pred_test_save= pred_test_save.reshape(y_length,-1).transpose(1,0)#to recover shape length,size -> size, lenght
	real_test_save = scaler.inverse_transform(seq_outOr_tst.squeeze().detach().numpy().reshape(-1,1))
	real_test_save = real_test_save.reshape(y_length,-1).transpose(1,0)

	#### FIRST REVERS SCALE TRANSFORMATION

	err = RMSE(real_test_save,pred_test_save)
	maerr = MAE(real_test_save,pred_test_save)
	ratioPeaks = NUMPEAK(real_test_save,pred_test_save, qTest90)


	writetofile(os.path.join(save_path,"bestV_pred.txt"),pred_test_save)
	writetofile(os.path.join(save_path,"bestV__real.txt"),real_test_save)
	writeErrResult(os.path.join(save_path,"bestV__test_rmse.txt"),err) # y_length
	writeErrResult(os.path.join(save_path,"bestV__test_mae.txt"),maerr)

	#Final log
	print(f'Test loss: {loss_tst:10.4f} {err_avg:10.4f} Diff Max-Min :{diffMaxMin:10.4f} Std: {stdPreds:2.4f}')


	#Global log - among runs
	type_rnn = args.type_rnn + "_unguided"
	return loss_tst, err_avg, stdPreds, ratioPeaks





#modelType	 epochs	inputDim	 xLength	 yLength	 learningRate	 unitsHiddenLayer	 typeTraining	 trainingSize	miniBatchSize	 typeAF	 lossTraint	 lossTest	
#RMSE	DiffMaxMin	stdDev	BestPrevIteration	BestLoss	BestStd	ratioPeaks	 folder	Comments	preproFile

dfResults = pd.read_csv(os.path.join(args.save_path,args.type_rnn,incident_group+"/LogResultsTests.csv"))


print(dfResults.columns)
###### ITERATE OVER ALL THE FOLDERS OF FINAL RESULTS
for index, row in dfResults.iterrows():
	folder = row['folder']
	bestIter = row['BestPrevIteration']
	hidden_units = int(row["unitsHiddenLayer"])
	preproFile = str(row["preproFile"])
	x_length = int(row["xLength"])
	y_length = int(row["yLength"])
	modelType = str(row["modelType"])

	### SPLIT THE TRAIN, TEST AND VAL WITH THAT FILE
	
	if preproFile == 'nan' or modelType!='lstm_guided':
	#if np.isnan(preproFile):
		continue
	df = pd.read_csv(preproFile, delimiter=',')

	# stop test of 100
	all_data = df['Response Time'].values.astype(float)#[:500]
	scaler = MinMaxScaler()#feature_range=(-1, 1)
	seq_inOr_tst, seq_outOr_tst, qVal50, test_size, qTest90 = splitData(all_data, scaler)

	bestStd = 0.0
	save_path = os.path.join(args.save_path,args.type_rnn,incident_group,folder)
	if np.isnan(bestIter):
		#Look for the folder and open Log.csv
		
		dfLogExp = pd.read_csv(os.path.join(save_path,"Log.csv"))

		bestLossVal = float('Inf')
		for idLogExp, rowLogExp in dfLogExp.iterrows():
			#Epoch	lossTrain	lossVal	stdDev or standarDeviation
			if 'stdDev' in dfLogExp.columns:
				stdev = rowLogExp["stdDev"]
			else:
				stdev = rowLogExp["standarDeviation"]

			if float(rowLogExp["lossVal"])< bestLossVal and float(stdev)>= qVal50:
				bestLossVal = float(rowLogExp["lossVal"])
				bestIter = int(rowLogExp["Epoch"])
				bestStd = float(stdev)
	else:
		bestIter = int(bestIter)

	### CREATE THE MODEL WITH THOS UNITS AND LOAD THE MODEL
	if args.type_rnn == 'rnn':
		model = RNN(x_dim, y_dim, hidden_units, args.type_actifun)
	else:#lstm
		model = LSTM(x_dim, y_dim, hidden_units, args.type_actifun)

	#If we did not find a better solution
	if np.isnan(bestIter):
		continue
	print(index, folder, preproFile, modelType, bestStd,qVal50, test_size, qTest90)

	fn = os.path.join(save_path,'vrnn_state_dict_'+str(bestIter)+'.pth')
	model.load_state_dict(torch.load(fn))

	loss_function = nn.MSELoss()
	BestLossTest, BestRMSE, BestStdTest, ratioPeaksTest = testModel(seq_inOr_tst, seq_outOr_tst, test_size, qTest90, save_path)

	dfResults["BestPrevIteration"].loc[index] = bestIter
	dfResults["BestLoss"].loc[index] = round(BestRMSE,0)
	dfResults["BestStd"].loc[index] = round(BestStdTest,4)
	dfResults["ratioPeaks"].loc[index] = round(ratioPeaksTest,4)

dfResults.to_csv(os.path.join(args.save_path,args.type_rnn,incident_group+"/LogResultsTests.csv"), index=False)




