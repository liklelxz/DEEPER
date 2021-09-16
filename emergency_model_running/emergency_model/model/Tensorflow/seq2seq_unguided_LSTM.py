import os
#os.environ['TP_CPP_MIN_LOG_LEVEL'] = '2'
#--
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys # for debugging
import copy
import random
import datetime

from preprocessor import *
from fileprocessor import *
from calculateError import *
'''
import preprocessing.fileprocessor
import preprocessing.preprocessor
import preprocessing.calculateError
'''
from modelGraph import *

from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes


# obtaining the Data ------------------------

import argparse

parser = argparse.ArgumentParser(description='lstm')
parser.add_argument('--fileName', type=str, default='',
                    help='Path where the files are located')
parser.add_argument('--logFileName', type=str, default='LogParamsError.csv',
                    help='Path where the log file is located')
parser.add_argument('--path', type=str, default='./../Results/lstm', 
                    help='Path where the files are located')
parser.add_argument('--x-length', type=int, default=14, metavar='XL',
                    help='previous time steps (default: 20)')
parser.add_argument('--y-length', type=int, default=7, metavar='YL',
                    help='steps to predict (default: 10)')
parser.add_argument('--minEpoch', type=int, default=500, metavar='ME',
                    help='minimum number of epochs (default: 20)')
parser.add_argument('--hiddenLayer', type=int, default=10, metavar='HL',
                    help='number of hidden layers (default: 20)')
parser.add_argument('--numLayers', type=int, default=1, metavar='NL',
                    help='number of layers')
parser.add_argument('--modelPath', type=str, default='', 
                    help='Path to restore the model from')
parser.add_argument('--learningRate', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--trainPercentage', type=float, default=0.75, metavar='TP',
                    help='training percentage')
parser.add_argument('--inputCols', type=str, default='C', 
                    help='columns to take as features')
parser.add_argument('--input-dim', type=int, default=1, metavar='ID',
                    help='steps to predict (default: 10)')
parser.add_argument('--output-dim', type=int, default=1, metavar='OD',
                    help='steps to predict (default: 10)')
parser.add_argument('--tanhOut', type=int, default=0, metavar='TL',
                    help='if to include or not tanh layer')
parser.add_argument('--randomSamp', type=int, default=0, metavar='OD',
                    help='not overlapping windows assigned randomly to training and testing set')
#parser.add_argument('--relu', type=int, default=0, metavar='RE', help='if to include or not relu layer')
parser.add_argument('--print_every', type=int, default=10, metavar='NL',help='number of layers')
#parser.add_argument('--decaySteps', type=int, default=1000, metavar='DS',
#                    help='decay steps')
#parser.add_argument('--decayRate', type=float, default=0.5, metavar='DR',
#                    help='decay rate')
parser.add_argument('--opt', type=str, default='Adam', 
                    help='Optimizer')

args = parser.parse_args()

x_length = args.x_length # the input sequence length
y_length = args.y_length # the output sequence length
fileName = args.fileName
path = args.path
minEpoch = args.minEpoch
hidden_size = args.hiddenLayer # LSTM hidden unit size
modelPath = args.modelPath
initial_learning_rate = args.learningRate # learning rate parameter

building = args.fileName.split("/")[-1][:-4] #output file name


assert len(args.inputCols) == args.input_dim

percentage = args.trainPercentage #0.75 # the percentage of data used for training
input_dim = args.input_dim # the number of input signals
output_dim = args.output_dim # the number of output signals

name = building+"_"+str(x_length)+"_"+str(y_length) # the name flag for the test case

folder  = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
save_path_name = os.path.join(args.path,building,folder)
os.makedirs(save_path_name)#os.getcwd() # the pwd to current directory


save_object_name = name + "_model" #name_flag # the state name to be saved
print "Model path ", args.modelPath,args.opt#, "Decay steps and rate", args.decaySteps,args.decayRate,args.opt

X_train_data, Y_train_data, X_test_data, Y_test_data = getData(args.fileName,x_length,y_length,percentage, args.input_dim,args.inputCols)#, args.randomSamp)

X_train = np.array(X_train_data)
Y_train = np.array(Y_train_data)
X_test = np.array(X_test_data)
Y_test = np.array(Y_test_data)

writeErrResult(os.path.join(save_path_name,name+"_X_Train.txt"),X_train[:,:,0])
writeErrResult(os.path.join(save_path_name,name+"_Y_Train.txt"),Y_train[:,:,0])
writeErrResult(os.path.join(save_path_name,name+"_X_Test.txt"),X_test[:,:,0])
writeErrResult(os.path.join(save_path_name,name+"_Y_Test.txt"),Y_test[:,:,0])

print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

#-----------------------------------------------------
# un-guided training method
#loss_t = 300 # needs to be some random value less than LOSS_LIMIT
ep = 0;
avg_rmse_lim = 3
LOSS_LIMIT = avg_rmse_lim * avg_rmse_lim
CONTINUE_FLAG = True
EPOCH_LIMIT = args.minEpoch
MIN_EPOCH_LIM = args.minEpoch

iteration = ["iteration"]
train_loss = ["trainLoss"]
train_RMSE = ["trainRMSE"]
test_loss = ["testLoss"]
train_RMSE_avg = ["trainRMSE-avg"] 
test_RMSE_avg = ["testRMSE-avg"]
test_rel0 = ["testRel0"]
test_rel1 = ["testRel1"]
past_loss_values = []
epoch_range = 5

#---------- RESTORING SAVED MODEL ----------------#
rnn_model =graph(args,feed_previous=True) #un-guided training model
temp_saver = rnn_model['saver']()

if args.modelPath == '':
    print("No previous model used")
    #tf.reset_default_graph()
    init = tf.global_variables_initializer()
# -------------- TRAINING ---------------#
with tf.Session() as sess:
    if args.modelPath != '':
        temp_saver.restore(sess, os.path.join(args.modelPath, save_object_name))#name+'_model'
    else:
        init.run()
    while CONTINUE_FLAG:
        #for i in range(y_length):
        #    print "dim dec_in ", Y_train[:,i].reshape(-1,input_dim).shape
        feed_dict = {       rnn_model['enc_inp'][t]:X_train[:,t].reshape(-1,input_dim) for t in range(x_length)         }
        feed_dict.update({  rnn_model['target_seq'][t]:Y_train[:,t,0].reshape(-1,output_dim) for t in range(y_length)   })
        #feed_dict.update({  rnn_model['dec_inp_init'][t]:Y_train[:,t].reshape(-1,input_dim) for t in range(y_length)    })
        #for t in range(y_length):
        #    feed_dict.update({  rnn_model['target_seq'][t]:Y_train[:,t,0].reshape(-1,output_dim) })

        train_t,loss_tr,out_t = sess.run([rnn_model['train_op'],rnn_model['loss'],rnn_model['reshaped_outputs']],feed_dict)

        if ep % args.print_every == 0:
            temp_output = np.reshape(out_t,(y_length,-1))
            temp_output = temp_output.transpose()
            temp_y_found = temp_output.tolist()
            temp_err = RMSE(Y_train_data[:,:,0],temp_y_found)

            train_loss.append(loss_tr)
            train_RMSE.append(temp_err)
            iteration.append(ep)
            rmse_avg = np.round(np.mean(temp_err),2)
            train_RMSE_avg.append(rmse_avg)

            #-------------------- STATE LOGGER--------------------------------
            # log state of identified values
            if ep % (EPOCH_LIMIT/5) == 0:
                temp_saver = rnn_model['saver']()
                save_path = temp_saver.save(sess,os.path.join(save_path_name,save_object_name))                
            #-----------------------------------------------------------------

            feed_dict2 = {rnn_model['enc_inp'][t]:X_test[:,t].reshape(-1,input_dim) for t in range(x_length)}
            #Y_temp = np.zeros((len(X_test),y_length), dtype=np.float)
            feed_dict2.update({rnn_model['target_seq'][t]:Y_test[:,t].reshape(-1,output_dim) for t in range(y_length)})
            loss_tst, out_t = sess.run([rnn_model['loss'],rnn_model['reshaped_outputs']],feed_dict2)
            matrix = np.reshape(out_t,(y_length,-1))
            matrix = matrix.transpose()
            Y_found = matrix.tolist()
            err = RMSE(Y_test[:,:,0],Y_found)
            listErrDay0, listErrDay1, avgERR0, avgERR1 = RelE(Y_found,Y_test[:,:,0],args.y_length)

            rmse_avg_test = np.round(np.mean(err),4)
            test_RMSE_avg.append(rmse_avg_test)
            test_loss.append(loss_tst)
            test_rel0.append(avgERR0)
            test_rel1.append(avgERR1)

        #if ep % (EPOCH_LIMIT/10) == 0:
            print ep,"Train loss:", loss_tr, "Train rmse:",rmse_avg, "Test loss:",loss_tst,"Testing RMSE/RelE0/RelE1:", rmse_avg_test, np.round(np.mean(avgERR0),2), np.round(np.mean(avgERR1),2)#, sess.run(tf.train.get_global_step())

        #-- condition to stop training - condition to keep track of past losses
        if ep < epoch_range:
            past_loss_values.append(loss_tr)
        else:
            past_loss_values.pop(0)
            past_loss_values.append(loss_tr)
        # increase the epoch count
        ep += 1
        #-- find if the entire range of previous losses are below a threshold
        count = 0
        for val in past_loss_values:
            if val < LOSS_LIMIT:
                count += 1
        #-- stopping condition for training
        if (count >= epoch_range or ep >= EPOCH_LIMIT) and ep>= MIN_EPOCH_LIM:
            print(count,ep)
            CONTINUE_FLAG = False

    print "--- training complete ---"

    save_path = temp_saver.save(sess,os.path.join(save_path_name,save_object_name))
    print "--- session saved ---"

    print "--- testing started ---"
    feed_dict2 = {rnn_model['enc_inp'][t]:X_test[:,t].reshape(-1,input_dim) for t in range(x_length)}
    Y_temp = np.zeros((len(X_test),y_length), dtype=np.float)
    feed_dict2.update({rnn_model['target_seq'][t]:Y_temp[:,t].reshape(-1,output_dim) for t in range(y_length)})
    loss_t, out_t = sess.run([rnn_model['loss'],rnn_model['reshaped_outputs']],feed_dict2)
    #(time,batch,dim)
    matrix = np.reshape(out_t,(y_length,-1))
    matrix = matrix.transpose()
    Y_found = matrix.tolist()
    err = RMSE(Y_test[:,:,0],Y_found)
    maerr = MAE(Y_test[:,:,0],Y_found)
    rmse_avg_test = np.round(np.mean(err),4)
    
    listErrDay0, listErrDay1, avgERR0, avgERR1 = RelE(Y_found,Y_test[:,:,0],args.y_length)
    print " loss: ", loss_t
    print "test error: ",err
    std_dev_1 = np.std(Y_found,axis=0)
    print "std dev of timestep 1:",std_dev_1

with open(os.path.join(args.path,args.logFileName),'a') as f:
    csv_reader = csv.writer(f, delimiter=',')
    csv_reader.writerow([folder,args.inputCols,str(args.input_dim),str(args.x_length)+"_"+str(args.y_length),args.fileName.split("/")[-1][:-4],args.learningRate, args.hiddenLayer, args.minEpoch, args.numLayers, args.randomSamp, rmse_avg_test, std_dev_1[0], avgERR0, avgERR1,args.opt])#,str(args.decaySteps)+"_"+str(args.decayRate),args.opt])

#------------- SAving files in date foler and directly in lstm/ -----------#
writeColumns(os.path.join(save_path_name,name+"_Train_loss_rmse.csv"), zip(iteration,train_loss,test_loss,train_RMSE_avg,test_RMSE_avg))
writeColumns(os.path.join(save_path_name,name+"_Train_rmse_STEP.csv"),zip(iteration,train_RMSE)) # saving the train RMSE values for every 200th epoch
writeErrResult(os.path.join(save_path_name,name+"_test_rmse.txt"),err) # y_length
#writeErrResult(os.path.join(args.path,name+"_test_rmse.txt"),err)
writeErrResult(os.path.join(save_path_name,name+"_test_mae.txt"),maerr)
#writeErrResult(os.path.join(args.path,name+"_test_mae.txt"),maerr)
writeErrResult(os.path.join(save_path_name,name+"_test_loss.txt"),[loss_t])
#writeErrResult(os.path.join(save_path_name,name+"_test_rel0.txt"),listErrDay0)
#writeErrResult(os.path.join(save_path_name,name+"_test_rel1.txt"),listErrDay1)


#np.save(os.path.join(save_path_name,name+"_pred"),Y_found)
#np.save(os.path.join(save_path_name,name+"_real"),Y_test_data)
writetofile(os.path.join(save_path_name,name+"_pred.txt"),np.round(Y_found,2))
writetofile(os.path.join(save_path_name,name+"_real.txt"),Y_test[:,:,0])

#writetofile(os.path.join(args.path,name+"_pred.txt"),np.round(Y_found,2))
#writetofile(os.path.join(args.path,name+"_real.txt"),Y_test[:,:,0])

print "----- run complete-------"
print datetime.datetime.now()    
    
    
    
