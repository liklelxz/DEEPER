import os
import argparse
import pickle 
import math
# import comet_ml in the top of your file
#from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from model.rnnPred import RNNPred

#from preprocessing.preproRNN import *

from preprocessing.calculateError import *
from preprocessor import *
from preprocessing.fileprocessor import *


from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FormatStrFormatter
import matplotlib.cm as cm
import datetime
import csv

####### CODE TAKEN FROM https://github.com/emited/VariationalRecurrentNeuralNetwork.git
parser = argparse.ArgumentParser(description='The Embedded Topic Model')
### data and file related arguments
parser.add_argument('--data_path', type=str, default='./Data/Water/', help='Path where the files are located')
parser.add_argument('--save_path', type=str, default='./Results/RNN/joint/', help='path to save results')
parser.add_argument('--modelArch', type=str, default='vrnn', help='path to save results')
#parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training')
parser.add_argument('--x-length', type=int, default=24, metavar='XL', help='previous time steps (default: 20)')
parser.add_argument('--y-length', type=int, default=12, metavar='XL', help='previous time steps (default: 20)')
parser.add_argument('--trainPercentage', type=float, default=0.75, metavar='TP', help='training percentage')
parser.add_argument('--numEpochs', type=int, default=100, metavar='XL', help='previous time steps (default: 20)')
parser.add_argument('--hiddenLayer', type=int, default=100, metavar='HL',help='number of hidden layers (default: 20)')
parser.add_argument('--numLayers', type=int, default=1, metavar='NL',help='number of layers')
parser.add_argument('--print_every', type=int, default=10, metavar='NL',help='number of layers')

parser.add_argument('--learningRate', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--use-relu', action='store_true', help='If to use a RELU or Sigmoid activation function. Default False')
parser.add_argument('--use-cuda', action='store_true',help='use cuda (default: False)')
parser.add_argument('--typeScale', type=str, default='mm', help='path to save results')
parser.add_argument('--inputCols', type=str, default='EW',  help='columns to take as features')
parser.add_argument('--input-dim', type=int, default=1, metavar='ID',help='steps to predict (default: 10)')
parser.add_argument('--output-dim', type=int, default=1, metavar='OD', help='steps to predict (default: 10)')
#typeGauss

args = parser.parse_args()
percentage = args.trainPercentage
typeScale = args.typeScale
x_length = args.x_length
y_length = args.y_length
x_dim = args.input_dim
y_dim = args.output_dim
clip = 1

#print_every = int(args.numEpochs/(args.numEpochs/20))
save_every = int(args.numEpochs/2)
X_train_data, Y_train_data, X_test_data, Y_test_data, scaler = getData(args.data_path,x_length,y_length,percentage, x_dim,args.inputCols,typeScale, 0)

print(X_train_data.shape, Y_train_data.shape, X_test_data.shape, Y_test_data.shape)

X_train_data = np.concatenate([X_train_data,Y_train_data[:,:-1,:]], axis=1)

print(X_train_data.shape, X_train_data.mean(), Y_train_data.mean())
building = args.data_path.split("/")[-1][:-4]# args.data_path[-6:-4]#building.csv #args.building
folder  = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
save_path = os.path.join(args.save_path,building,folder)
os.makedirs(save_path)

print(save_path)

setSize = len(X_train_data)
setSizeTest = len(Y_test_data)

if typeScale=='mm':
    YreTrainScale = scaler.inverse_transform(np.reshape(Y_train_data,(setSize*y_length,y_dim)))
    YreTrainScale = np.reshape(YreTrainScale,(setSize,y_length,y_dim))

    YreTestScale = scaler.inverse_transform(np.reshape(Y_test_data,(setSizeTest*y_length,y_dim)))
    YreTestScale = np.reshape(YreTestScale,(setSizeTest,y_length,y_dim))
elif typeScale =='zz':
    aux1=np.reshape(Y_train_data,(setSize*y_length,y_dim))
    YreTrainScale = aux1 *scaler[1] + scaler[0]
    YreTrainScale = np.reshape(YreTrainScale,(setSize,y_length,y_dim))

    aux2 = np.reshape(Y_test_data,(setSizeTest*y_length,y_dim))
    YreTestScale = aux2 * scaler[1] + scaler[0]
    YreTestScale = np.reshape(YreTestScale,(setSizeTest,y_length,y_dim))
else:
	print("no scaler")


setSizeY_train = len(Y_train_data)
setSizeY_test = len(Y_test_data)

def train(epoch, model):
	dataX = Variable(torch.from_numpy(X_train_data).double())#.transpose(0, 1)
	dataY = Variable(torch.from_numpy(Y_train_data).double())


	predictions, hn = model(dataX.transpose(0, 1), setSizeY_train, y_length)#,dataY.transpose(0, 1)
	#print("Train ",len(predictions), hn.shape)
	
	mseLoss = nn.MSELoss()
	mse_loss = mseLoss(predictions[-y_length:,:,0], dataY[:,:,0].transpose(0, 1))

	#print("training pred vs datay ",predictions.shape, dataY.shape)
	#mse_loss = torch.mean((predictions[-y_length:,:,0] - dataY[:,:,0].transpose(0, 1))**2)

	mse_loss.backward()
	nn.utils.clip_grad_norm_(model.parameters(), clip)
	optimizer.step()
	#grad norm clipping, only in pytorch version >= 1.10


	



	# Returning to not-scaled values to measure RMSE

	predictions = predictions.transpose(1,0).detach().numpy() #l,b,d -> b, l, d


	if typeScale=='mm':

		YreTrainScale = scaler.inverse_transform(np.reshape(Y_train_data,(setSizeY_train*y_length,y_dim)))
		YreTrainScale = np.reshape(YreTrainScale,(setSizeY_train,y_length,y_dim))

		predictions = scaler.inverse_transform(np.reshape(predictions,(setSizeY_train*y_length,y_dim)))
		predictions = np.reshape(predictions,(setSizeY_train,y_length,y_dim))

	elif typeScale =='zz':

		aux1=np.reshape(Y_train_data,(setSizeY_train*y_length,y_dim))
		YreTrainScale = aux1 *scaler[1] + scaler[0]
		YreTrainScale = np.reshape(YreTrainScale,(setSizeY_train,y_length,y_dim))

		aux2 = np.reshape(predictions,(setSizeY_train*y_length,y_dim))
		predictions = aux2 * scaler[1] + scaler[0]
		predictions = np.reshape(predictions,(setSizeY_train,y_length,y_dim))
	else:
		#print("no scaler")
		YreTrainScale = Y_train_data 


	rmse1 = RMSE(YreTrainScale[:,:, 0], predictions[:,:,0])

	rmse1_orig = np.round(np.mean(rmse1), 2)


	#if (epoch % args.print_every == 0):
		#print('Train Epoch: {}\tLoss: {:.6f}\trmse1: {:.6f}\trmse2: {:.6f}'.format( epoch, mse_loss.item() ,  rmse1_orig.item(), rmse2_orig.item() ))

	return  epoch, mse_loss.item() ,  rmse1_orig.item()
	#return 'Train Epoch: {:5d}\tLossE: {:.4f}\tLossW: {:.4f}\trmse1: {:.4f}\trmse2: {:.4f}'.format( epoch, mse_lossE.item(), mse_lossW.item() ,  rmse1_orig.item(), rmse2_orig.item() )

def test(epoch, model):
	"""uses test data to evaluate  likelihood of the model"""


	dataX = Variable(torch.from_numpy(X_test_data[:,:x_length,:]).transpose(0, 1))#B,L,D -> L,B,D
	dataY = Variable(torch.from_numpy(Y_test_data).transpose(0, 1))

	predictions_in, hn = model(dataX, setSizeY_test, x_length-1)


	#The last prediction of this function is the first prediction, if we just do x-length-1. The last prediction is x-length
	lastPred = predictions_in[-1,None,:,:]
	#Or the previous to the last real value from x-length. Actually no, because I need hn
	#lastPred = dataX[:,x_length,:]

	
	predictions = model.sample(lastPred, args.y_length, hn)

	predictions = torch.cat(predictions)


	mse_loss = torch.mean((predictions[:,:,0] - dataY[:,:,0])**2)


	predictions = predictions.transpose(1,0).detach().numpy()

	if typeScale=='mm':

		YreTestScale = scaler.inverse_transform(np.reshape(Y_test_data,(setSizeY_test*y_length,y_dim)))
		YreTestScale = np.reshape(YreTestScale,(setSizeY_test,y_length,y_dim))

		predictions = scaler.inverse_transform(np.reshape(predictions,(setSizeY_test*y_length,y_dim)))
		predictions = np.reshape(predictions,(setSizeY_test,y_length,y_dim))

	elif typeScale =='zz':

		aux1=np.reshape(Y_test_data,(setSizeY_test*y_length,y_dim))
		YreTestScale = aux1 *scaler[1] + scaler[0]
		YreTestScale = np.reshape(YreTestScale,(setSizeY_test,y_length,y_dim))

		aux2 = np.reshape(predictions,(setSizeY_test*y_length,y_dim))
		predictions = aux2 * scaler[1] + scaler[0]
		predictions = np.reshape(predictions,(setSizeY_test,y_length,y_dim))
	else:
		print("no scaler")
		YreTestScale = Y_test_data
	

	rmse1 = RMSE(YreTestScale[:,:, 0], predictions[:,:,0])

	rmse1_orig = np.round(np.mean(rmse1), 2)


	#print(predictions.shape)
	numPred1 = predictions[:,:,0]

	#numPred1 = pred1.detach().numpy()
	#numPred2 = pred2.detach().numpy()
	print(Y_test_data.mean(), YreTestScale.mean(), numPred1.mean())
	step=0
	#print( numPred1.shape, YreTestScale.shape)
	Energy = np.concatenate([ numPred1[:,step,None], YreTestScale[:,step, 0,None]],axis=1 )

	plt.plot(Energy) # (len, dim)
	plt.savefig("{}/e{}_energy".format(save_path, epoch), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
	plt.clf()

	#print('====================> Test rmse: {:.4f}, {:.4f} '.format( rmse1_orig, rmse2_orig))
	return mse_loss.item(),   rmse1_orig, rmse1
	#return 'Test rmse: {:.4f}, {:.4f} '.format( rmse1_orig, rmse2_orig)


model = RNNPred(x_dim, args.numLayers, args.hiddenLayer).double()



#model_parameters = filter(lambda p: p.requires_grad, model.parameters())

totalVarPar = 0 
model_parameters = filter(lambda p: p.requires_grad, model.parameters())

for val in model.state_dict():
	print(val)
#for val in model.state_dict():
for val in model_parameters:
	shape = val.shape
	print(shape, len(shape))
	variable_parameters = 1
	for dim in shape:
	    variable_parameters *= dim
	totalVarPar += variable_parameters
print('Total number of parameters',totalVarPar)

#print(model.state_dict()['rnn.weight_hh_l0'])



'''
rnn.weight_ih_l0
rnn.weight_hh_l0
rnn.bias_ih_l0
rnn.bias_hh_l0
fc.weight
fc.bias
'''
optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate)

for epoch in range(args.numEpochs):
	epoch, loss_tr, rmse1_orig = train(epoch, model)

	#saving model
	if epoch % args.print_every == 0:
		loss_tst, rmse1_orig_tst, rmse1Arr = test(epoch, model)
		
		fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
		torch.save(model.state_dict(), fn)
		#print('Saved model to '+fn)

		print('Train Epoch: {:5d}\tLoss: {:.4f}\trmse1: {:.4f}'.format( epoch,  loss_tr,  rmse1_orig), 
				'Test rmse: {:.4f} '.format( rmse1_orig_tst))


loss_tst, rmse1_orig_tst,rmse1Arr = test(epoch, model)

fn = os.path.join(save_path,'vrnn_state_dict_'+str(epoch)+'.pth')
torch.save(model.state_dict(), fn)
#print('Saved model to '+fn)

print('Loss - rmse: {:.4f}, {:.4f} '.format( loss_tst, rmse1_orig_tst))

#writing log, results
with open(os.path.join(args.save_path,building+"/LogResults.csv"),'a') as f:
    csv_reader = csv.writer(f, delimiter=',')
    #lstm
    #csv_reader.writerow([args.inputCols,str(args.input_dim),str(args.x_length)+"_"+str(args.y_length),args.fileName.split("/")[-1][:-4],args.learningRate, args.hiddenLayer, args.minEpoch, args.numLayers, args.randomSamp, rmse_avg_test, avgERR0, avgERR1])
    #csv_reader.writerow(["Building","Type", "x-len", "y-len",  "learning_rate", "layers", "hiddenUnits", "epochs","Training loss", "Test loss","RMSE energy","RMSE water","Folder"])
    csv_reader.writerow([args.inputCols,x_dim,str(args.x_length)+"_"+str(args.y_length),building,args.learningRate, args.hiddenLayer, args.numEpochs, args.numLayers,'0',  rmse1_orig_tst ,str(round(loss_tst,3)),folder ])


weights_h = model.state_dict()['rnn.weight_hh_l0'].detach().numpy()
#depending on how many fc there is fc.weight or fc.0 or fc.2
weights_o = model.state_dict()['fc.2.weight'].detach().numpy()

# Plot the matrix
plt.matshow(weights_h, cmap='viridis')
plt.savefig("{}/W_h".format(save_path), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
plt.clf()

plt.matshow(weights_o, cmap='viridis')
plt.savefig("{}/W_o".format(save_path), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)

plt.clf()

fn = os.path.join(args.save_path,'vrnn_state_dict_'+str(epoch)+'.pth')
torch.save(model.state_dict(), fn)
print('Saved model to '+fn)


#------------- SAving files in date foler and directly in lstm/ -----------#
#writeColumns(os.path.join(save_path_name,building+"_Train_loss_rmse.csv"), zip(iteration,train_loss,test_loss,train_RMSE_avg,test_RMSE_avg))
#writeColumns(os.path.join(save_path_name,building+"_Train_rmse_STEP.csv"),zip(iteration,train_RMSE)) # saving the train RMSE values for every 200th epoch
writeErrResult(os.path.join(save_path,building+"_test_rmse.txt"),rmse1Arr) # y_length

#writeErrResult(os.path.join(save_path_name,building+"_test_mae.txt"),maerr)

#writeErrResult(os.path.join(save_path_name,building+"_test_loss.txt"),[loss_t])


