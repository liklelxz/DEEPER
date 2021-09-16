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
from preprocessing.preprocessorPeriods import *
from preprocessing.fileprocessor import *
from modelPytorch import *


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

parser.add_argument('--folder', type=str, default='', metavar='AF',help='Folder to test')
parser.add_argument('--bestIter', type=str, default='', metavar='BI',help='Iteration with std greater than quantile and smallest error')
parser.add_argument('--type_group', type=str, default='law', metavar='TG',help='TypeGroup')
parser.add_argument('--periods', type=int, default=8, metavar='PE',help='number of in which is devided the dataset 17 semesters or 8 year')


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
periods = args.periods
hidden_units = args.hidden_layer
mini_batch_size = args.mini_batch
epochs = args.epochs
type_training = args.type_training
learning_rate = args.learning_rate
incident_group = args.type_group
save_path = os.path.join(args.save_path,args.type_rnn,incident_group)

def getData(x_length, y_length, periods, sampling_method):
	sets, stats, indexRef = getPreProcessedData(incident_group[:3], x_length, y_length, periods, sampling_method)
	#train_normalized, valid_normalized, test_normalized, scaler, qTra50,qTra90, qVal50,  qVal90 , qTest90 = preprocessing(file_name, percentage)
	trainX, valX, testX, trainY, valY, testY = sets
	scaler , qTra50, qTra90, qVal50, qVal90, qTest50, qTest90  = stats
	

	print("Quantiles 50Train ", qTra50, " qTrain90: ",qTra90)

	train_in = torch.FloatTensor(trainX)
	train_out = torch.FloatTensor(trainY)#create_inout_sequences(train_data_normalized, x_length, y_length)
	val_in, val_out		= torch.FloatTensor(valX), 	torch.FloatTensor(valY)#create_inout_sequences(valid_data_normalized, x_length, y_length)
	test_in, test_out 	= torch.FloatTensor(testX), torch.FloatTensor(testY)#create_inout_sequences(test_data_normalized, x_length, y_length)


	sizeTrain = len(train_in) 
	sizeTest = len(test_in)
	if type_training=='AD':
		mini_batch_size = sizeTrain
	num_batches = int(sizeTrain/mini_batch_size) 


	allIdx = range(sizeTrain)

	seq_inOr_tra  = train_in.transpose(0,1)[:,:,None]
	seq_outOr_tra = train_out.transpose(0,1)[:,:,None]


	val_size = len(val_in)
	print(val_size)
	seq_inOr_val = val_in.transpose(0,1)[:,:,None]#torch.Size([15, 1334, 1])
	seq_outOr_val = val_out.transpose(0,1)[:,:,None]

	test_size = len(test_in)
	seq_inOr_tst = test_in.transpose(0,1)[:,:,None]#torch.Size([15, 1334, 1])
	seq_outOr_tst = test_out.transpose(0,1)[:,:,None]

	return seq_inOr_tra, seq_inOr_val, seq_inOr_tst, seq_outOr_tra, seq_outOr_val, seq_outOr_tst, sizeTrain, val_size, test_size, qTra50, qTra90,  qVal50, qTest90, scaler, indexRef




def testing(seq_inOr, seq_outOr, qTra50, qTra90,  size, scaler,  indexRef):
	#model.eval()
	list_preds = []
	loss_test_batch = 0

	dfGroup, idxTestX, idxTestY =  indexRef

	#TEST BATCH GUIDED

	####### REVIEW IF IT BATCH AND INDIVIDUAL TESTING GIVE SAME RESULTS, THEY DID
	test_preds = []
	with torch.no_grad():


		hidden = (torch.zeros(1, size, model.hidden_layer_size), torch.zeros(1, size, model.hidden_layer_size))
		preds_batch=model(seq_inOr,seq_outOr, size,hidden, y_length)
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
			preds=model(seq_inOr[:,i,:], seq_outOr[:,i,:] , 1,hidden, y_length)

			#preds=model(seq_in,1,hiddien, y_length)
			single_loss = loss_function(preds, seq_outOr[:,i,:,None])
			
			#####    single_loss = single_loss + torch.sum(seq_outOr)


			loss_test += single_loss

		#Adding as many lists as steps
		for step in range(y_length):
			test_preds.append(preds[step].detach().numpy().reshape((y_dim)))

		list_preds.append(test_preds)
	#print(preds[step].detach().numpy().squeeze().shape, np.asarray(list_preds).shape)

	predsMetric = np.transpose(np.asarray(list_preds),[1,0,2])


	#### CALCULATING MAE, MSE TO SAVE THEM TO FILE
	pred_test_save = scaler.inverse_transform(predsMetric.squeeze().reshape(-1,1))#to flat the data
	pred_test_save= pred_test_save.reshape(y_length,-1).transpose(1,0)#to recover shape length,size -> size, lenght
	real_test_save = scaler.inverse_transform(seq_outOr.squeeze().detach().numpy().reshape(-1,1))
	real_test_save = real_test_save.reshape(y_length,-1).transpose(1,0)

	### Ploting back into the time series
	step_plot = 0
	idxStepX = idxTestX[:,step_plot]
	idxStepY = idxTestY[:,step_plot]
	dfGroup['Response Time'].iloc[idxStepX] = seq_inOr_tst[step_plot].detach().numpy().squeeze()
	dfGroup['Response Time'].iloc[idxStepY] = seq_outOr_tst[step_plot].detach().numpy().squeeze()
	dfGroup['Pred Response'] = 0
	dfGroup['Pred Response'].iloc[idxStepY] = pred_test_save[:,step_plot]

	#plt.plot(np.concatenate((pred_test[step_plot],seq_outOr_tst[step_plot]),axis =1))
	dfGroup[['Response Time','Pred Response']].plot()
	plt.savefig("{}/PredvsReal_TestRealSort2".format(save_path), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
	plt.clf()


	#### FIRST REVERS SCALE TRANSFORMATION
	err = RMSE(real_test_save,pred_test_save)
	maerr = MAE(real_test_save,pred_test_save)
	ratioPeaks50 = NUMPEAK(real_test_save,pred_test_save, scaler.inverse_transform(np.array([qTra50]).reshape(-1,1)))
	ratioPeaks90 = NUMPEAK(real_test_save,pred_test_save, scaler.inverse_transform(np.array([qTra90]).reshape(-1,1)))

	ratioNoPeaks50 = NUM_NOPEAK(real_test_save,pred_test_save, scaler.inverse_transform(np.array([qTra50]).reshape(-1,1)))
	ratioNoPeaks90 = NUM_NOPEAK(real_test_save,pred_test_save, scaler.inverse_transform(np.array([qTra90]).reshape(-1,1)))



	writetofile(os.path.join(save_path,"bestV_pred.txt"),pred_test_save)
	writetofile(os.path.join(save_path,"bestV__real.txt"),real_test_save)
	writeErrResult(os.path.join(save_path,"bestV__test_rmse.txt"),err) # y_length
	writeErrResult(os.path.join(save_path,"bestV__test_mae.txt"),maerr)

	return loss_test/size, np.transpose(np.asarray(list_preds),[1,0,2]), loss_test_batch.item(), lossBoxPlot, np.mean(err), ratioPeaks50, ratioPeaks90, ratioNoPeaks50, ratioNoPeaks90




def testModel(seq_inOr_tst, seq_outOr_tst, qTra50, qTra90, test_size, save_path,typeSplit,scaler, indexRef):
	loss_test, pred_test, loss_tst_batch , loss_box_tst, err_avg, \
	ratioPeaks50,ratioPeaks90, ratioNoPeaks50, ratioNoPeaks90 = testing(seq_inOr_tst, seq_outOr_tst, qTra50, qTra90, test_size, scaler, indexRef)

	loss_test = loss_test.item()
	diffTestPerStep = np.max(pred_test,axis=1) -  np.min(pred_test,axis=1)
	diffTestPerStep = np.round(diffTestPerStep,3)
	maxDiffTest = np.max(diffTestPerStep)
	stdPredsPerStepTest = np.round(np.std(pred_test,axis=1),4)
	stdPredsTotalTest = np.std(pred_test).round(3)

	stdPredsTotalTestOrig = scaler.inverse_transform(np.array([stdPredsTotalTest]).reshape(-1,1))

	'''
	diffMaxMin = pred_test.max() - pred_test.min()
	stdPreds = np.std(pred_test)
	loss_tst = loss_tst.item()#to bring from graph network space
	'''

	step_plot = 0
	plt.plot(np.concatenate((pred_test[step_plot],seq_outOr_tst[step_plot]),axis =1))
	plt.savefig("{}/bestV_PredvsReal_{}".format(save_path, typeSplit), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
	plt.clf()

	#Plot boxplot of losses
	dfBox = pd.DataFrame(loss_box_tst)
	dfBox.boxplot()
	plt.savefig(os.path.join(save_path,'bestV_boxplot'))
	plt.clf()

	'''
	#### CALCULATING MAE, MSE TO SAVE THEM TO FILE
	pred_test_save = scaler.inverse_transform(pred_test.squeeze().reshape(-1,1))#to flat the data
	pred_test_save= pred_test_save.reshape(y_length,-1).transpose(1,0)#to recover shape length,size -> size, lenght
	real_test_save = scaler.inverse_transform(seq_outOr_tst.squeeze().detach().numpy().reshape(-1,1))
	real_test_save = real_test_save.reshape(y_length,-1).transpose(1,0)

	#### FIRST REVERS SCALE TRANSFORMATION

	err = RMSE(real_test_save,pred_test_save)
	maerr = MAE(real_test_save,pred_test_save)
	ratioPeaks = NUMPEAK(real_test_save,pred_test_save, qTra90)
	'''


	#Final log
	print(f'Test loss: {loss_test:10.4f} {err_avg:10.4f} Diff Max-Min :{maxDiffTest:10.4f} Std: {stdPredsTotalTest:2.4f} Peaks: {ratioPeaks50:2.4f}')

	#Global log - among runs
	type_rnn = args.type_rnn + "_unguided"
	return loss_test, err_avg, stdPredsTotalTest, ratioPeaks50, ratioPeaks90, ratioNoPeaks50, ratioNoPeaks90, stdPredsTotalTestOrig





#modelType	 epochs	inputDim	 xLength	 yLength	 learningRate	 unitsHiddenLayer	 typeTraining	 trainingSize	miniBatchSize	 typeAF	 lossTraint	 lossTest	
#RMSE	DiffMaxMin	stdDev	BestPrevIteration	BestLoss	BestStd	ratioPeaks	 folder	Comments	preproFile

dfResults = pd.read_csv(os.path.join(args.save_path,args.type_rnn,incident_group+"/LogResults.csv"))




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
	periods = int(row["periods"])
	sampling_method = str(row["samplingMethod"])

	### SPLIT THE TRAIN, TEST AND VAL WITH THAT FILE

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

			if float(rowLogExp["lossVal"])< bestLossVal and float(stdev)>= qTra50:
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


	seq_inOr_tra, seq_inOr_val, seq_inOr_tst, seq_outOr_tra, seq_outOr_val, seq_outOr_tst, sizeTrain, val_size, test_size, \
	qTra50, qTra90, qVal50, qTest90, scaler, indexRef = getData(x_length, y_length, periods, sampling_method)

	#If we did not find a better solution
	if np.isnan(bestIter):
		continue
	print(index, folder, preproFile, modelType, bestStd,qVal50, test_size, qTest90)

	fn = os.path.join(save_path,'vrnn_state_dict_'+str(bestIter)+'.pth')
	#model = 
	modelDict = torch.load(fn)

	model.load_state_dict(modelDict)#['model_state_dict']


	##### CORRECT LOAD OF PARAMETERS


	model.eval()
	#
	loss_function = nn.MSELoss()


	### EVALUATING THE MODEL IN THE TESTING SET AND COUTING PEAKES greater than qTest90
	BestLossTest, BestRMSE, BestStdTest, ratioPeaksTest50, ratioPeaksTest90, ratioNoPeaks50, ratioNoPeaks90, \
	stdPredsTestOrig = testModel(seq_inOr_tst, seq_outOr_tst, qTra50, qTra90, test_size, save_path, "test", scaler, indexRef)

	#BestLossVal, BestValRMSE, BestStdVal, ratioPeaksVal = testModel(seq_inOr_val, seq_outOr_val, val_size, save_path, "valid")

	#model_parameters = zip(model.state_dict(), filter(lambda p: p.requires_grad, model.parameters())  )
	#print([(name, round(p.grad.data.norm(2).item(),4)) for name, p in model_parameters])

#testRMSEPrevVal	testLossPrevVal	testStdPrevVal	testPeaksPrevVal

	#dfResults["BestPrevIteration"].loc[index] = bestIter
	dfResults["testRMSEPrevVal"].loc[index] = round(BestRMSE,0)
	dfResults["testLossPrevVal"].loc[index] = round(BestLossTest,2)
	dfResults["testStdVal"].loc[index] = stdPredsTestOrig
	dfResults["testStdPrevVal"].loc[index] = round(BestStdTest,2)
	dfResults["testPeaksPrevVal50"].loc[index] = round(ratioPeaksTest50,4)
	dfResults["testPeaksPrevVal90"].loc[index] = round(ratioPeaksTest90,4)
	#testNoPeak50	testNoPeak90
	dfResults["testNoPeak50"].loc[index] = round(ratioNoPeaks50,4)
	dfResults["testNoPeak90"].loc[index] = round(ratioNoPeaks90,4)


dfResults.to_csv(os.path.join(args.save_path,args.type_rnn,incident_group+"/LogResults.csv"), index=False)




