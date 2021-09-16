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

parser.add_argument('--folder', type=str, default='', metavar='AF',help='Folder to test')
parser.add_argument('--bestIter', type=str, default='', metavar='BI',help='Iteration with std greater than quantile and smallest error')


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
hidden_units = args.hidden_layer
type_training = args.type_training
learning_rate = args.learning_rate
incident_group = args.data_path.split("/")[-1][:-4]# args.data_path[-6:-4]#building.csv #args.building
folder  = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
save_path = os.path.join(args.save_path,args.type_rnn,incident_group,folder)
print(save_path)

timestamp_folder = args.folder
model_path = os.path.join(args.save_path,args.type_rnn,incident_group,timestamp_folder)
print(model_path)

os.makedirs(save_path)



#Reading and spliting
df = pd.read_csv(file_name, delimiter=',')







# stop test of 100
all_data = df['Response Time'].values.astype(float)#[:500]









train_data_size = int(percentage*len(all_data))#431
validation_data_size = train_data_size + int(0.5*(1-percentage)*len(all_data))
train_data = all_data[:train_data_size]
valid_data = all_data[train_data_size : validation_data_size]
test_data = all_data[validation_data_size:]
print(len(train_data),len(valid_data), len(test_data))



#Preprocessing
scaler = MinMaxScaler()#feature_range=(-1, 1)
scaler.fit(train_data.reshape(-1, 1))
train_normalized = scaler.transform(train_data.reshape(-1, 1))
valid_normalized = scaler.transform(valid_data.reshape(-1,1))
test_normalized = scaler.transform(test_data.reshape(-1, 1))

print(np.quantile(train_normalized,0.9), np.quantile(valid_normalized,0.9), np.quantile(test_normalized,0.9))



#Creating containers or pytorch variables
train_data_normalized = torch.FloatTensor(train_normalized).view(-1)
valid_data_normalized = torch.FloatTensor(valid_normalized).view(-1)
test_data_normalized = torch.FloatTensor(test_normalized).view(-1)

qVal50=np.quantile(valid_normalized,0.5)
qVal90=np.quantile(valid_normalized,0.9)
qTest90=np.quantile(test_normalized,0.9)

print("Q-0.5: ",np.quantile(train_normalized,0.5), qVal50, np.quantile(test_normalized,0.5))
print("Q-0.9: ",np.quantile(train_normalized,0.9), qVal90, np.quantile(test_normalized,0.9))


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
		self.lstmEnc = nn.LSTM(input_size, hidden_layer_size)

		self.rnnDec = nn.RNN(input_size, hidden_layer_size)

		if type_actifun=='relu':
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.ReLU())
		else:#Sigmoid
			self.linear = nn.Sequential(nn.Linear(hidden_layer_size, output_size),  nn.Sigmoid())

	def forward(self, input_seq, output_seq, batch_size,hidden_cell, pred_length):

		#Encoder Forward
		lstm_outEnc, hidden_cellEnc = self.lstmEnc(input_seq.view(len(input_seq) ,batch_size, -1), hidden_cell)

		#Decoder Forward
		predictions = []


		dec_input = input_seq[-1]# + output_seq[:-1]
		
		#print(dec_input.shape, input_seq.shape)
		
		hidden_cellDec = hidden_cellEnc[0]#h,c
		for step in range(pred_length):
			lstm_outDec, hidden_cellDec = self.rnnDec(dec_input.view(1,batch_size, -1), hidden_cellDec)
			dec_input = self.linear(lstm_outDec)
			predictions.append(dec_input)
		#predictions = self.linear(lstm_out.view(-1, self.hidden_layer_size))

		#predictions = predictions.reshape(len(input_seq),batch_size, self.output_size)
		
		#print(torch.cat(predictions[-pred_length:]).shape)
		return torch.cat(predictions[-pred_length:])


if args.type_rnn == 'rnn':
	model = RNN(x_dim, y_dim, hidden_units, args.type_actifun)
else:#lstm
	model = LSTM(x_dim, y_dim, hidden_units, args.type_actifun)

fn = os.path.join(model_path+'/vrnn_state_dict_'+bestIter+'.pth')
model.load_state_dict(torch.load(fn))
#model.eval()


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

	return loss_test/size, np.transpose(np.asarray(list_preds),[1,0,2]), loss_test_batch.item(), lossBoxPlot, np.mean(err)



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



loss_val, pred_val, loss_val_batch , loss_box_val, err_val_avg= testing(seq_inOr_val, seq_outOr_val, val_size)


 

loss_tst, pred_test, loss_tst_batch , loss_box_tst, err_avg= testing(seq_inOr_tst, seq_outOr_tst, test_size)
diffMaxMin = pred_test.max() - pred_test.min()
stdPreds = np.std(pred_test)
loss_tst = loss_tst.item()#to bring from graph network space

diffMaxMin_all = []
stdPreds_all = []
for i in range(y_length):
	diffMaxMin_all.append(pred_test[i].max() - pred_test[i].min())
	stdPreds_all.append(np.std(pred_test[i]))

print(diffMaxMin_all)
print(stdPreds_all)

step_plot = 0
plt.plot(np.concatenate((pred_test[step_plot],seq_outOr_tst[step_plot]),axis =1))
plt.savefig("{}/_PredvsReal_Test".format(save_path), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
plt.clf()

#Plot boxplot of losses
dfBox = pd.DataFrame(loss_box_tst)
dfBox.boxplot()
plt.savefig(os.path.join(save_path+'/_boxplot_'))
plt.clf()

#### CALCULATING MAE, MSE TO SAVE THEM TO FILE
pred_test_save = scaler.inverse_transform(pred_test.squeeze().reshape(-1,1))#to flat the data
pred_test_save= pred_test_save.reshape(y_length,-1).transpose(1,0)#to recover shape length,size -> size, lenght
real_test_save = scaler.inverse_transform(seq_outOr_tst.squeeze().detach().numpy().reshape(-1,1))
real_test_save = real_test_save.reshape(y_length,-1).transpose(1,0)

#print(f'Test_pred: {pred_test_save[:,0]}')

#### FIRST REVERS SCALE TRANSFORMATION

err = RMSE(real_test_save,pred_test_save)
maerr = MAE(real_test_save,pred_test_save)


writetofile(os.path.join(save_path,incident_group+"_pred.txt"),pred_test_save)
writetofile(os.path.join(save_path,incident_group+"_real.txt"),real_test_save)
writeErrResult(os.path.join(save_path,incident_group+"_test_rmse.txt"),err) # y_length
writeErrResult(os.path.join(save_path,incident_group+"_test_mae.txt"),maerr)

#Final log
print(f'Test loss: {loss_tst:10.4f} {err_avg:10.4f} Diff Max-Min :{diffMaxMin:10.4f} Std: {stdPreds:2.4f}')

#flgPeaksReal = np.where(real_test_save>=qTest90, real_test_save)
#numPeaksReal = np.sum(flgPeaksReal,axis=1)

#flgPeaksPred = np.where(pred_test_save>=qTest90, pred_test_save)
#numPeaksPred = np.sum(flgPeaksPred,axis=1)
#ratioPeaks = 0

# If there are no peaks:
#if numPeaksReal != 0:
#	ratioPeaks = np.mean(numPeaksPred/numPeaksReal)


#Global log - among runs
type_rnn = args.type_rnn + "_unguided"

with open(os.path.join(args.save_path,args.type_rnn,incident_group+"/LogResults_Previous.csv"),'a') as f:
    csv_reader = csv.writer(f, delimiter=',')
    csv_reader.writerow([type_rnn, args.epochs,x_dim,str(args.x_length),str(args.y_length),args.learning_rate, args.hidden_layer, args.type_training, \
    						sizeTrain, args.mini_batch, args.type_actifun, 0, round(loss_tst,3), round(err_avg,0), round(diffMaxMin_all[0],2), round(diffMaxMin_all[1],2), round(diffMaxMin_all[2],2), round(diffMaxMin_all[3],2), round(diffMaxMin_all[4],2), round(diffMaxMin,2), \
    						round(stdPreds_all[0],4), round(stdPreds_all[1],4), round(stdPreds_all[2],4), round(stdPreds_all[3],4), round(stdPreds_all[4],4), round(stdPreds,4), folder, timestamp_folder,file_name, args.bestIter])