import numpy as np
import sys
from sklearn.metrics import mean_squared_error
#from pyramid.arima import auto_arima
from pmdarima.arima import auto_arima
from preprocessing.fileprocessor import *
from preprocessing.preprocessor import *
from preprocessing.calculateError import *
from preprocessing.preprocessorPeriods import *

import datetime
import os
import argparse

parser = argparse.ArgumentParser(description='arima')
parser.add_argument('--data_path', type=str, default='./Data/With_Features/fire.csv', help='Path where the files are located')
parser.add_argument('--save_path', type=str, default='./Results/arima/', help='path to save results')
parser.add_argument('--percentage', type=float, default=0.75, metavar='TP', help='training percentage. Rest is splitted in validation and testing sets')
parser.add_argument('--x-length', type=int, default=15, metavar='XL',
                    help='previous time steps (default: 20)')
parser.add_argument('--y-length', type=int, default=5, metavar='YL',
                    help='steps to predict (default: 10)')
parser.add_argument('--inputCols', type=str, default='R', 
                    help='columns to take as features')
parser.add_argument('--input-dim', type=int, default=1, metavar='ID',
                    help='steps to predict (default: 10)')
parser.add_argument('--validation', type=int, default=0, metavar='ID',
                    help='omit validation set (default: 0)')
parser.add_argument('--randomSamp', type=int, default=0, metavar='OD',
                    help='not overlapping windows assigned randomly to training and testing set')
parser.add_argument('--periods', type=int, default=8, metavar='PE',help='number of in which is devided the dataset 17 semesters or 8 year')
parser.add_argument('--sampling_method', type=str, default='Random', metavar='SA',help='Select the sampling method - random or from best fit distribution (Use: Dist)')

args = parser.parse_args()

"""def NUMPEAK(real_test_save, pred_test_save, qTest90):

	flgPeaksReal = np.where(real_test_save>=qTest90)
	numPeaksReal = np.sum(flgPeaksReal,axis=1)

	flgPeaksPred = np.where(pred_test_save>=qTest90)
	numPeaksPred = np.sum(flgPeaksPred,axis=1)

	#print("qTest90:",qTest90)
	#print("FReal:", flgPeaksReal,numPeaksReal)
	#print("FPred:", flgPeaksPred,numPeaksPred)

	ratioPeaks = np.mean(numPeaksPred/numPeaksReal)
	return ratioPeaks"""


#Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual,number_of_prediction):
	model = auto_arima(Actual, trace=False, error_action='ignore', suppress_warnings=True)
	model.fit(Actual)
	#prediction = model_fit.forecast()[0]
	forecast = model.predict(n_periods=number_of_prediction)
	#indexs = [0,1]
	#forecast = pd.DataFrame(forecast, index=indexs, columns=['Prediction'])
	forecast = [round(elem, 2) for elem in forecast]

	return forecast

'''
def rmse(y_hat,y):
	#Method 1
	#Get RMSE over each day
	rmse_days = []

	#y_hat = [item for sublist in y_hat for item in sublist]
	rmse_days = np.sqrt(mean_squared_error(y_hat, y, multioutput='raw_values'))
	#Get average RMSE
	rmse1 = sum(rmse_days)/len(rmse_days)
	rmse1 = round(rmse1,2)
	#Method 2
	#Get sum of squares per day
	sum_of_squares = sum((y_hat - y)**2)
	#sum over days, divide by (num_days * num_samples), take sqrt()
	rmse2 = np.sqrt(sum(sum_of_squares)/(y_hat.shape[1]*y_hat.shape[0]))
	rmse2 = round(rmse2,2)
	rmse_days = rmse_days.tolist()
	rmse_days = [round(x,2) for x in rmse_days]

	return rmse_days,rmse1,rmse2
'''
def arima_model():

	x_length = args.x_length
	y_length = args.y_length
	percentage = args.percentage
	periods = args.periods
	sampling_method = args.sampling_method


	folder = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
	incident_group = args.data_path.split("/")[-1][:-4]
	save_path = os.path.join(args.save_path,incident_group,folder)
	os.makedirs(save_path)
	print(save_path)

	dir_path = save_path + "/"

	real_value_path = dir_path + incident_group + "_"+str(x_length)+"_"+str(y_length)+"_real.txt"
	rela_error_path = dir_path + incident_group + "_"+str(x_length)+"_"+str(y_length)+"_relativeError.txt"
	xtrain_value_path = dir_path + incident_group + "_"+str(x_length)+"_"+str(y_length)+"_xtrain.txt"

	predicted_value_path = dir_path + incident_group + "_" + str(x_length) + "_" + str(y_length) + "_pred.txt"

	print("Running for:", incident_group)

	pred = open(predicted_value_path, "w")
	real = open(real_value_path, "w")
	xtrain = open(xtrain_value_path, "w")
	error = open(rela_error_path, "w")
	fh2 = open(dir_path + incident_group + "_"+str(x_length)+"_"+str(y_length)+"_test_rmse.txt", "w")
	fh3 = open(dir_path + incident_group + "_"+str(x_length)+"_"+str(y_length)+"_test_mae.txt", "w")
	param_file = open(dir_path+"params_history.txt","w")
	param_file.write("Building : "+incident_group+"\nxlength : "+str(x_length)+"\nyLength : "+str(y_length)+"\nColumns:"+str(args.inputCols)+"\n\n")
	param_file.close()


	#X_train, Y_train, X_test, Y_test, qTest90 = getData(args.data_path, x_length, y_length,percentage,args.input_dim,args.inputCols, args.validation)
	sets, stats, indexRef = getPreProcessedData(incident_group[:3], x_length, y_length, periods, sampling_method, "Arima")

	X_train, X_Val, X_test, Y_train, Y_Val, Y_test = sets
	scaler , qTra50, qTra90, qVal50, qVal90, qTest50, qTest90 = stats

	#response_times, X_train, Y_train, X_test, Y_test = getData(filename, x_length, y_length,percentage)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	predictions = []
	counter = 0
	for i in range(0, X_test.shape[0]):
		#real_value = X_test[i,:].reshape((-1,1))
		prediction = StartARIMAForecasting(X_test[i,:].reshape((-1,1)),y_length)
		for k in range(0, y_length):
			if prediction[k]<0:
				prediction[k] = 0
			pred.write(str(round(prediction[k], 2)) + ",")
			#real.write(str(round(real_value[k][0], 2)) + ",")
		pred.write("\n")
		#real.write("\n")
		#counter += 1
		predictions.append(prediction)

	for i in range(0, Y_test.shape[0]):
		real_value = Y_test[i, :].reshape((-1, 1))
		for k in range(0, y_length):
			real.write(str(round(real_value[k][0],6))+",")
		real.write("\n")

	for i in range(0, X_train.shape[0]):
		x_value = X_train[i, :].reshape((-1, 1))
		for k in range(0, x_length):
			xtrain.write(str(round(x_value[k][0],6))+",")
		xtrain.write("\n")		

	#predictions = [item for sublist in predictions for item in sublist]
	predictions = np.array(predictions).reshape((-1, y_length))

#Get RMSE over each day, and different rmse calculations
#Previous in-file function
#rmse_days,rmse1,rmse2 =  rmse(predictions,Y_test)
#pointing to calculateError.py

	rmse_days=  RMSE(predictions,Y_test)
	mae_days = MAE(Y_test, predictions)
	relative_error0, relative_error1, avg0,avg1 = RelE(predictions,Y_test,y_length)

	Y_test = np.array(Y_test).reshape((-1, y_length))

	ratioPeaks50 = NUMPEAK(Y_test, predictions, qTra50)
	ratioPeaks90 = NUMPEAK(Y_test, predictions, qTra90)

	ratioNOPeaks50 = NUM_NOPEAK(Y_test, predictions, qTra50)
	ratioNOPeaks90 = NUM_NOPEAK(Y_test, predictions, qTra90)
	diffMaxMin = predictions.max() - predictions.min()
	stdPreds = np.std(predictions)

	print (rmse_days)

	for r in rmse_days:
		fh2.write(str(r) + '\n')


	for r in mae_days:
		fh3.write(str(r) + '\n')
	error.write("RE0  , RE1  ,\n")
	for i in range(y_length):
		error.write(str(round(relative_error0[i],2))+" ,"+str(round(relative_error1[i],2))+"\n")
	error.write("\naverage0 : "+str(round(avg0,2))+"\naverage1 : "+str(round(avg1,2)))
	fh2.close()
	fh3.close()
	pred.close()
	real.close()
	xtrain.close()
	error.close()

	with open(os.path.join(args.save_path+"/LogResults.csv"),'a') as f:
		csv_reader = csv.writer(f, delimiter=',')
		csv_reader.writerow(['arima',str(args.x_length),str(args.y_length), round(np.mean(rmse_days),0), round(np.mean(mae_days),0), round(diffMaxMin,2), round(stdPreds,4), round(ratioPeaks50,4),round(ratioPeaks90,4), round(ratioNOPeaks50,4),round(ratioNOPeaks90,4), sampling_method, folder,"",args.data_path ])

arima_model()
