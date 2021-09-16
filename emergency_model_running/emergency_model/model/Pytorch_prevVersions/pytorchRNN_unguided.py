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

from modelPytorch import *
from preprocessing.calculateError import *
from preprocessing.preprocessorFuncs import *

from fileprocessor import *


pd.options.display.float_format = '{:.4f}'.format
np.set_printoptions(suppress=True)
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
parser.add_argument('--print_every', type=int, default=20, metavar='NL',help='number of layers')


clip = 0.01

args = parser.parse_args()

#Setting parameters
percentage = args.percentage
file_name = args.data_path
x_length = args.x_length
y_length = args.y_length
x_dim = args.input_dim
y_dim = args.output_dim
mini_batch_size = args.mini_batch
epochs = args.epochs
hidden_units = args.hidden_layer
type_training = args.type_training
learning_rate = args.learning_rate
print_every = args.print_every
incident_group = args.data_path.split("/")[-1][:-4]# args.data_path[-6:-4]#building.csv #args.building
folder  = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
save_path = os.path.join(args.save_path,args.type_rnn,incident_group,folder)
os.makedirs(save_path)
print(save_path)

#Global log - among runs
type_rnn = args.type_rnn + "_unguided"



# stop test of 100
#all_data = df['Response Time'].values.astype(float)#[:500]


train_normalized, valid_normalized, test_normalized, scaler, qTra50, qVal50,  qVal90 , qTest90 = preprocessing(file_name, percentage)


#Creating containers or pytorch variables
train_data_normalized = torch.FloatTensor(train_normalized).view(-1)
valid_data_normalized = torch.FloatTensor(valid_normalized).view(-1)
test_data_normalized = torch.FloatTensor(test_normalized).view(-1)



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



train_in, train_out = create_inout_sequences(train_data_normalized, x_length, y_length)
val_in, val_out = create_inout_sequences(valid_data_normalized, x_length, y_length)
test_in, test_out = create_inout_sequences(test_data_normalized, x_length, y_length)

sizeTrain = len(train_in) 
sizeTest = len(test_in)
if type_training=='AD':
	mini_batch_size = sizeTrain
num_batches = int(sizeTrain/mini_batch_size) 




if args.type_rnn == 'rnn':
	model = RNN(x_dim, y_dim, hidden_units, args.type_actifun)
else:#lstm
	model = LSTM(x_dim, y_dim, hidden_units, args.type_actifun)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


allIdx = range(sizeTrain)
#Transform lists of x-y in arrays and move length to first dimension: torch.Size([len, batch_size, 1])
seq_inOr_tra  = torch.stack(train_in).transpose(0,1)[:,:,None]
seq_outOr_tra = torch.stack(train_out).transpose(0,1)[:,:,None]

val_size = len(val_in)
print(val_size)
seq_inOr_val = torch.stack(val_in).transpose(0,1)[:,:,None]#torch.Size([15, 1334, 1])
seq_outOr_val = torch.stack(val_out).transpose(0,1)[:,:,None]

test_size = len(test_in)
seq_inOr_tst = torch.stack(test_in).transpose(0,1)[:,:,None]#torch.Size([15, 1334, 1])
seq_outOr_tst = torch.stack(test_out).transpose(0,1)[:,:,None]

def NUMPEAK(real_test_save, pred_test_save, qTest90):

	flgPeaksReal = np.where(real_test_save>=qTest90)
	numPeaksReal = np.sum(flgPeaksReal,axis=1)

	flgPeaksPred = np.where(pred_test_save>=qTest90)
	numPeaksPred = np.sum(flgPeaksPred,axis=1)

	ratioPeaks = np.mean(numPeaksPred/numPeaksReal)
	return ratioPeaks



def getIndexes(i):
	'''
	Depending on the training type, we send fixed indexes, random or the indexes of all the dataset
	'''
	if type_training=='FB':
		idx = allIdx[i*mini_batch_size  : (i+1)*mini_batch_size]
	elif type_training == 'RB':
		idx = random.choices(allIdx, k=mini_batch_size)
	else:#'AD'
		assert (1==num_batches)
		idx = allIdx
	return idx


def training(seq_inOr, seq_outOr):
	model.train()
	lossEpoch = 0
	for i in range(num_batches):
		idx = getIndexes(i)
		seq_in = seq_inOr[:,idx,:]
		seq_out = seq_outOr[:,idx,:]

		size = seq_in.shape[1]

		#for seq, labels in train_inout_seq:
		optimizer.zero_grad()
		hidden_cell = (torch.zeros(1, size, model.hidden_layer_size),
		               torch.zeros(1, size, model.hidden_layer_size))

		y_pred = model(seq_in, seq_out, size, hidden_cell, y_length)

		#y_pred = torch.stack(y_pred)
		#print(y_pred.shape,seq_out.shape)
		single_loss = loss_function(y_pred, seq_out)

		#####  single_loss = single_loss + torch.sum(seq_out)
		
		lossEpoch += single_loss


		single_loss.backward()



		#not sure when tu use, before or after step
		#nn.utils.clip_grad_norm_(model.parameters(), clip)



		optimizer.step()

	return lossEpoch/num_batches, y_pred

def testing(seq_inOr, seq_outOr, size):
	model.eval()
	list_preds = []
	loss_test_batch = 0

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

	#### FIRST REVERS SCALE TRANSFORMATION
	err = RMSE(real_test_save,pred_test_save)
	maerr = MAE(real_test_save,pred_test_save)
	ratioPeaks = NUMPEAK(real_test_save,pred_test_save, qTest90)

	return loss_test/size, np.transpose(np.asarray(list_preds),[1,0,2]), loss_test_batch.item(), lossBoxPlot, np.mean(err), ratioPeaks


colsLog = ['lossTrain','lossVal','stdDev']

dfLog = pd.DataFrame(columns=colsLog)

#[args.type_rnn, x_length,args.y_length,learning_rate, hidden_units, type_training, args.type_actifun]+\
#modelType	 epochs	inputDim	 xLenght	 yLenght	 learningRate	 unitsHiddenLayer	 typeTraining	 trainingSize	miniBatchSize	 typeAF
colsLog = ['modelType','xLength','yLength','learningRate','unitsHiddenLayer','typeTraining','typeAF'] + colsLog

for i in range(1,y_length+1):
	colsLog.append('diffMaxMinVal'+str(i))
for i in range(1,y_length+1):
	colsLog.append('stdPerStepVal'+str(i))
colsLog.append('lossTest')
colsLog.append('stdTest')
for i in range(1,y_length+1):
	colsLog.append('diffMaxMinTest'+str(i))
for i in range(1,y_length+1):
	colsLog.append('stdPerStepTest'+str(i))


dfLogPerIter = pd.DataFrame(columns=colsLog)


#Printing parameters and number
totalVarPar = 0 
model_parameters = zip(model.state_dict(), filter(lambda p: p.requires_grad, model.parameters())  )
for name, val in model_parameters:#######zip(model.state_dict(),model_parameters):
	shape = val.shape
	print(name, shape, len(shape))
	variable_parameters = 1
	for dim in shape:
	    variable_parameters *= dim
	totalVarPar += variable_parameters
print('Total number of parameters',totalVarPar)


bestLosss = float("Inf")#torch.Tensor([])
thresholdStd = float(qVal50)#torch.Tensor([float(qVal50)])
bestEpoch = 0
bestLossVal = float('Inf')
bestStd = 0


for e in range(epochs):
	loss_train, pred_train = training(seq_inOr_tra, seq_outOr_tra)
	loss_train = loss_train.item()
	if e==0 or e%(epochs/print_every) == 0 or e==epochs-1:

		#To analyze norms
		#for p in model_parameters### req_grad and grad, difference ?? list(filter(lambda p: p.grad is not None, model.parameters())):
		#	print(p.grad.data.norm(2).item())

		model_parameters = zip(model.state_dict(), filter(lambda p: p.requires_grad, model.parameters())  )
		

		#Printing distribution of norms
		#print([(name, round(p.grad.data.norm(2).item(),4)) for name, p in model_parameters])


		####### Results for the VALIDATION SET
		loss_val, pred_val, loss_val_batch , loss_box, err_val, numPeaksVal = testing(seq_inOr_val, seq_outOr_val, val_size)
		loss_val = loss_val.item()
		diffValPerStep = np.max(pred_val,axis=1) -  np.min(pred_val,axis=1)
		diffValPerStep = np.round(diffValPerStep,3)
		maxDiffVal = np.max(diffValPerStep)
		stdPredsPerStepVal = np.round(np.std(pred_val,axis=1),4)
		stdPredsTotalVal = np.std(pred_val).round(3)

		#Keep track of best results for validation
		if err_val< bestLosss and thresholdStd <stdPredsTotalVal:
			bestLosss = err_val
			bestEpoch = e
			bestStd = stdPredsTotalVal


		##### Results for the TESTING SET
		loss_test, pred_test, loss_test_batch, loss_box_test, err_test, numPeaksTest = testing(seq_inOr_tst, seq_outOr_tst, test_size)
		loss_test = loss_test.item()
		diffTestPerStep = np.max(pred_test,axis=1) -  np.min(pred_test,axis=1)
		diffTestPerStep = np.round(diffTestPerStep,3)
		maxDiffTest = np.max(diffTestPerStep)
		stdPredsPerStepTest = np.round(np.std(pred_test,axis=1),4)
		stdPredsTotalTest = np.std(pred_test).round(3)


		print(f'epoch: {e:6} Train loss: {loss_train:10.6f} Validation loss: {loss_val:10.6f} {err_val:10.6f} MaxDiff Max-Min Val: {maxDiffVal:5.4f} Std: {stdPredsTotalVal:2.2f}')
		#Shapes:{pred_val.shape, seq_outOr_val.shape}: len, batch,1
		
		#print(loss_train.item(),loss_val.item())
		#print([loss_train,loss_val, stdPredsTotalVal]+diffValPerStep.squeeze().tolist()+stdPredsPerStepVal.squeeze().tolist()+[loss_test]+[stdPredsTotalTest]+diffTestPerStep.tolist()+stdPredsPerStepTest.tolist())
		dfLog.loc[e] = [loss_train,loss_val, stdPredsTotalVal]
		dfLogPerIter.loc[e] = [args.type_rnn, x_length,args.y_length,learning_rate, hidden_units, type_training, args.type_actifun]+\
								[loss_train,loss_val, stdPredsTotalVal]+diffValPerStep.squeeze().tolist()+stdPredsPerStepVal.squeeze().tolist()+\
								[loss_test]+[stdPredsTotalTest]+diffTestPerStep.squeeze().tolist()+stdPredsPerStepTest.squeeze().tolist()  

		#Plot pred vs real
		step_plot = 0
		plt.plot(np.concatenate((pred_val[step_plot],seq_outOr_val[step_plot]),axis =1))
		plt.savefig("{}/PredvsReal_{}".format(save_path,e), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
		plt.clf()

		#Plot boxplot of losses
		dfBox = pd.DataFrame(loss_box )
		dfBox.boxplot()
		plt.savefig(os.path.join(save_path+'/boxplot_'+str(e)))
		plt.clf()

		## SAVING THE MODEL
		fn = os.path.join(save_path+'/vrnn_state_dict_'+str(e)+'.pth')
		torch.save(model.state_dict(), fn)


print("Best iteration: ", bestEpoch, bestLosss)

loss_tst, pred_test, loss_tst_batch , loss_box_tst, err_avg , numPeaksTest = testing(seq_inOr_tst, seq_outOr_tst, test_size)
diffMaxMin = pred_test.max() - pred_test.min()
stdPreds = np.std(pred_test)
loss_tst = loss_tst.item()#to bring from graph network space


step_plot = 0
plt.plot(np.concatenate((pred_test[step_plot],seq_outOr_tst[step_plot]),axis =1))
plt.savefig("{}/PredvsReal_Test".format(save_path), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
plt.clf()

#Plot boxplot of losses
dfBox = pd.DataFrame(loss_box_tst)
dfBox.boxplot()
plt.savefig(os.path.join(save_path+'/boxplotTest'))
plt.clf()

#### CALCULATING MAE, MSE TO SAVE THEM TO FILE
pred_test_save = scaler.inverse_transform(pred_test.squeeze().reshape(-1,1))#to flat the data
pred_test_save= pred_test_save.reshape(y_length,-1).transpose(1,0)#to recover shape length,size -> size, lenght
real_test_save = scaler.inverse_transform(seq_outOr_tst.squeeze().detach().numpy().reshape(-1,1))
real_test_save = real_test_save.reshape(y_length,-1).transpose(1,0)

#### FIRST REVERS SCALE TRANSFORMATION

err = RMSE(real_test_save,pred_test_save)
maerr = MAE(real_test_save,pred_test_save)


writetofile(os.path.join(save_path,incident_group+"_pred.txt"),pred_test_save)
writetofile(os.path.join(save_path,incident_group+"_real.txt"),real_test_save)
writeErrResult(os.path.join(save_path,incident_group+"_test_rmse.txt"),err) # y_length
writeErrResult(os.path.join(save_path,incident_group+"_test_mae.txt"),maerr)

#Final log
print(f'Final: {e:3} Train loss: {loss_train:10.4f} Test loss: {loss_tst:10.4f} {err_avg:10.4f} Diff Max-Min :{diffMaxMin:10.4f} Std: {stdPreds:2.4f}')

#Local log - iteration log
dfLog.index.name = 'Epoch'
dfLog.to_csv(os.path.join(save_path+'/Log.csv'))

dfLogPerIter.index.name = 'Epoch'
dfLogPerIter.to_csv(os.path.join(args.save_path,args.type_rnn,incident_group+"/LogResultsPerIter.csv"), mode='a', header=False)



#Coppy and paste for the HEADER
#modelType, epochs,inputDim, xLenght, yLenght, learningRate, unitsHiddenLayer, typeTraining, trainingSize,miniBatchSize, typeAF, 
#lossTrain, lossTest, RMSEtest, diffMaxMinTest, stdDevTest, numPeaksTest, 
# lossRMSEValFinal	stdValFinal	numPeaksValFinal	BestPrevIteration	BestLoss	BestStd	ratioPeaks
#folder, Comments
with open(os.path.join(args.save_path,args.type_rnn,incident_group+"/LogResults.csv"),'a') as f:
    csv_reader = csv.writer(f, delimiter=',')
    csv_reader.writerow([type_rnn, args.epochs,x_dim,str(args.x_length),str(args.y_length),args.learning_rate, args.hidden_layer, args.type_training, \
    						sizeTrain, args.mini_batch, args.type_actifun, round(loss_train,3), round(loss_tst,3), round(err_avg,0), round(diffMaxMin,2), round(stdPreds,4), numPeaksTest,  \
    						round(err_val), round(stdPredsTotalVal), numPeaksVal, \
    						bestEpoch, bestLosss, bestStd, numPeaksVal ,folder,"",file_name ])
