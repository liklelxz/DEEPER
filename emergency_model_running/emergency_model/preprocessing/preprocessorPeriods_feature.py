import numpy as np
import pandas as pd
import matplotlib
import itertools as it
#import h5py
import random
import os
from os import path
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import scipy.stats
import scipy.stats as st
import statsmodels as sm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

listDateRange17 = [('2011-05','2011-11'),('2011-11','2012-05'),('2012-05','2012-11'),('2012-11','2013-05'),
                 ('2013-05','2013-11'),('2013-11','2014-05'),('2014-05','2014-11'),('2014-11','2015-05'),
                 ('2015-05','2015-11'),('2015-11','2016-05'),('2016-05','2016-11'),('2016-11','2017-05'),
                 ('2017-05','2017-11'),('2017-11','2018-05'),('2018-05','2018-11'),('2018-11','2019-05'),
                 ('2019-05','2020-01')]

listDateRange8 = [('2011-05','2012-05'),('2012-05','2013-05'),('2013-05','2014-05'),('2014-05','2015-05'),
                 ('2015-05','2016-05'),('2016-05','2017-05'),('2017-05','2018-05'),('2018-05','2020-01')]

'''
listPermu =[('train', 'validation', 'test'),
            ('validation', 'test', 'train'),
            ('test', 'train', 'validation'),
            ('train', 'test', 'validation'),
            ('validation', 'train', 'test'),
            ('test', 'validation', 'train'),
             ]
'''
listPermu =[('train', 'validation', 'test'),
            ('validation', 'test', 'train'),
            ('test', 'train', 'validation'),
            ('train', 'validation', 'test'),
            ('validation', 'test', 'train'),
            ('test', 'train', 'validation'),
             ]

listSplits = ['train','validation','test']

trainPercentage = 0.5
validPercentage = 0.25
testPercentage  = 0.25
#x_length = 0
#y_length = 0
#folder = ""

'''
fire_best_fit_params = ('nct',0.948558475720322, 2.5551844740981386, -0.10947086507084537, 46.73392810881188)
law_best_fit_params = ('nct',1.4805583427751872, 2.572263218415913, -0.060215006913632796, 18.579087455717563)
structural_best_fit_params = ('mielke',6.175189935823845, 0.9783516376708281, -12.584593748359062, 27.768010394741083)
utility_best_fit_params = ('fatiguelife',1.2636659977397333, -18.310204666293618, 342.2274920228938)
'''

fire_best_fit_params = ('nct',0.954992160922538, 2.5607707503861086, -0.13566981829479977, 45.758028912950515)
law_best_fit_params = ('nct',1.4841020732096917, 2.6335319343288255, -0.02664023592103516, 18.000852909325275)
structural_best_fit_params = ('invgamma',0.9835580939342791, -25.15789088682314, 174.1199903234571)
utility_best_fit_params = ('fatiguelife',1.2717683647346891, -19.021760376513893, 343.0353665685458)


def createFolder(listDateRange, groupType, sampling_method, periods):

    save_path = 'Data/MixedPeriods/'
    #dateLenghtFolder = listDateRange[0][0]+'_'+listDateRange[0][1]+'_'+str(x_length)+'_'+str(y_length)+'_'+sampling_method
    
    dateLenghtFolder = listDateRange[0][0]+'_'+listDateRange[0][1]+'_'+str(periods)+'_'+sampling_method
    
    folder = save_path + dateLenghtFolder + '/'

    if path.exists(folder):
        print("Existing folder " ,folder)

        '''
        hf = h5py.File(folder+groupType+'.h5', 'r')

        print(list(hf.keys()))
        newDFTrainX = hf['trainX']
        newDFTrainY = hf['trainY']
        newDFValX = hf['validX']
        newDFValY = hf['validY']
        newDFTestX = hf['testX']
        newDFTestY = hf['testY']
        q50Train = hf['quantiles'][0]
        q90Train = hf['quantiles'][1]
        q50Val = hf['quantiles'][2]
        q90Val = hf['quantiles'][3]
        q50Test = hf['quantiles'][4]
        q90Test = hf['quantiles'][5]
        scaler = hf['scaler']

        return folder, (newDFTrainX, newDFValX,newDFTestX, newDFTrainY,  newDFValY, newDFTestY, scaler, q50Train, q90Train, q50Valid, q90Valid, q50Test, q90Test)
        '''
    else:
        print("Creating folder", folder)
        os.makedirs(folder)
        

    return folder, None
    

def minMaxData(train_data, valid_data, test_data):
    #Preprocessing
    
    scaler = MinMaxScaler()#feature_range=(-1, 1)
    scaler.fit(train_data)
    train_normalized = scaler.transform(train_data.reshape(-1, 1))
    valid_normalized = scaler.transform(valid_data.reshape(-1,1))
    test_normalized = scaler.transform(test_data.reshape(-1, 1))

    return train_normalized, valid_normalized, test_normalized, scaler


def create_inout_sequences(input_data, xw, yw):
    '''
    Input: Sequence or dataFrame of one column ('Response Time')
    Output: lists of X and list of Y
    '''
    in_seq = []
    out_seq = []

    idx_in = []
    idx_out = []

    L = len(input_data)
    for i in range(L-xw-yw):
        train_seq = input_data[i:i+xw]
        train_label = input_data[i+xw:i+xw+yw]

        idx_in.append(input_data[i:i+xw].index)
        idx_out.append(input_data[i+xw:i+xw+yw].index)
        in_seq.append(train_seq )
        out_seq.append(train_label)

    #print(in_seq,out_seq)
    return in_seq, out_seq, idx_in, idx_out

def splitTraining(all_data, listSplit, dictSplit):
    '''
    Input: dataframe, could be more than one column
    Output: lists of X and list of Y
    '''
    
    dictDFrames = {}
    #the subRange in the period, if access with loc has positions from 0 to shape[0]
    prevIdx = 0 #all_data.index[0]
    originalSize = all_data.shape[0]
    for setSplit in listSplit:
        endIdx = int(dictSplit[setSplit]*originalSize)
        
        dictDFrames[setSplit] = all_data.iloc[prevIdx: prevIdx + endIdx]
        
        #print(prevIdx,prevIdx + endIdx, dictDFrames[setSplit].shape[0])
        
        prevIdx = prevIdx + endIdx
        

    return dictDFrames

def createResponseTimeColumn(EMOriginalRT):
	EMOriginalRT["Creation Date"]=pd.to_datetime(EMOriginalRT["Creation Date"])
	EMOriginalRT = EMOriginalRT.sort_values(by=["Creation Date"])
	EMOriginalRT["Response Time"] = (pd.to_datetime(EMOriginalRT["Closed Date"]) - pd.to_datetime(EMOriginalRT["Creation Date"])).dt.total_seconds()
	EMOriginalRT["minute"] = 60
	EMOriginalRT["Response Time"] = EMOriginalRT["Response Time"].div(EMOriginalRT["minute"])
	EMOriginalRT = EMOriginalRT.drop(["minute"], axis=1)
	return EMOriginalRT


# CREATE GROUP TYPE WITH FIRST 3 LETTERS OF INCIDENT TYPE
# VERIFY IF THERE IS NO POSSIBILITY FOR DIFFERENT GROUPS TO HAVE THE SAME 3 INITIAL LETTERS
# Split per group type
def splitGroupTypes(EMOriginalRT):
	EMOriginalRT['Group Type'] = EMOriginalRT['Incident Type'].str[:3]
	listGroupsNames = EMOriginalRT.groupby('Group Type').groups.keys()
	#Converted to list to access the index when processing only one group
	listGroupsNames = list(listGroupsNames)
	listDfPerGroup = []
	for groupStr in listGroupsNames:
	    #print(groupStr)
	    groupDF = pd.DataFrame(EMOriginalRT[(EMOriginalRT['Group Type'] == groupStr)])
	    listDfPerGroup.append(groupDF)

	return listDfPerGroup, EMOriginalRT, listGroupsNames


def getSplitInPeriod(dfGroup, listPermu, dictSplit, listDateRange):

	trainList = []
	valList = []
	testList = []

	countPermuSplits = 0

	# Get the split along all periods
	for iniDate, endDate in listDateRange:
		dfInRange = dfGroup.loc[(dfGroup["Creation Date"]>=iniDate)  & (dfGroup["Creation Date"]<endDate)]
		dictDFrames = splitTraining(dfInRange, listPermu[countPermuSplits%3], dictSplit)

		tra = dictDFrames['train']
		val = dictDFrames['validation']
		tst = dictDFrames['test']

		#Appending in one list all the pieces from different periods
		#print(dfInRange.shape[0],tra.shape[0], tra.index[:3], val.shape[0], tst.shape[0])
		trainList.append(tra)
		valList.append(val)
		testList.append(tst)

		countPermuSplits +=1
	    
	return trainList, valList, testList

def replaceValues(validPointsTrain, dfGroup, q90Train, groupType, sampling_method):

    print('Outliers in Train: ', validPointsTrain.loc[(validPointsTrain['Response Time']>q90Train)].shape[0], 'Null: ',  validPointsTrain.loc[(validPointsTrain["Response Time"].isna()==True)].shape[0] )

    # Selecting indexes of outliers or missing values
    outliers = dfGroup.loc[(dfGroup['Response Time']>q90Train) | (dfGroup["Response Time"].isna()==True)]
    # Selecting valid points only from training set
    validPoints = validPointsTrain.loc[(validPointsTrain['Response Time']<=q90Train)]
    minVal = min(validPoints['Response Time'])

    if sampling_method == 'Random':
        newValues = random.choices(list(validPoints['Response Time']), k=outliers.shape[0])
    else:
        if groupType == 'Fir':
            newValues = st.nct.rvs(fire_best_fit_params[1],fire_best_fit_params[2],fire_best_fit_params[3],fire_best_fit_params[4],outliers.shape[0])
        elif groupType == 'Law':            
            newValues = st.nct.rvs(law_best_fit_params[1],law_best_fit_params[2],law_best_fit_params[3],law_best_fit_params[4],outliers.shape[0])
        elif groupType == 'Str':
            newValues = st.invgamma.rvs(structural_best_fit_params[1],structural_best_fit_params[2],structural_best_fit_params[3],outliers.shape[0])
        elif groupType == 'Uti':
            newValues = st.fatiguelife.rvs(utility_best_fit_params[1],utility_best_fit_params[2],utility_best_fit_params[3],outliers.shape[0])
        else:
            print("For other incident types, use the script in Analysis/Distribution folder to get best fit distribution")
            newValues = None

    # Some distributions might include values that are too big. To maintain them smaller than the outliers, we replace them with that value
    newValues = np.asarray(newValues)
    newValues[newValues > q90Train] = q90Train
    numMinVals = newValues[newValues <= 0].shape[0]
    newValues[newValues <= 0]= minVal

    dfGroup['Response Time'].iloc[outliers.index] = newValues

    print('Replaced: ', len(newValues),' Outliers: ', dfGroup.loc[(dfGroup['Response Time']>q90Train)].shape[0],\
             'Null: ',  dfGroup.loc[(dfGroup["Response Time"].isna()==True)].shape[0], " Negative: " , numMinVals)
    return outliers, validPoints, dfGroup



def transformXY(concatenatedDFs, dfGroup, columns, x_length, y_length):
    newDFX = []
    newDFY = []
    newIdX = []
    newIdY = []
    for batch in concatenatedDFs:
        #print(dfGroup[columns].iloc[batch.index])
        batchTransformedX, batchTransformedY, batchIdX, batchIdY  = create_inout_sequences(dfGroup[columns].iloc[batch.index] , x_length, y_length)
        #In order for later be able to concatenate 0,0 with m,n
        if len(batchTransformedX)!=0:
            newDFX.append(batchTransformedX)
            newDFY.append(batchTransformedY)
            newIdX.append(batchIdX)
            newIdY.append(batchIdY)

    return newDFX, newDFY, newIdX, newIdY
    

def saveHistogramPlots(groupType, newDFTrainX, newDFValX,newDFTestX, newDFTrainY,  newDFValY, newDFTestY, folder):
    # Save the histograms of the distributions
    plt.hist(newDFTrainX.reshape(-1,1))
    plt.savefig("{}{}_trainX".format(folder,groupType), bbox_inches='tight')
    plt.clf()
    
    plt.hist(newDFValX.reshape(-1,1))
    plt.savefig("{}{}_valX".format(folder,groupType), bbox_inches='tight')
    plt.clf()

    plt.hist(newDFTestX.reshape(-1,1))
    plt.savefig("{}{}_testX".format(folder,groupType), bbox_inches='tight')
    plt.clf()

    plt.hist(newDFTrainY.reshape(-1,1))
    plt.savefig("{}{}_trainY".format(folder,groupType), bbox_inches='tight')
    plt.clf()
    
    plt.hist(newDFValY.reshape(-1,1))
    plt.savefig("{}{}_valY".format(folder,groupType), bbox_inches='tight')
    plt.clf()

    plt.hist(newDFTestY.reshape(-1,1))
    plt.savefig("{}{}_testY".format(folder,groupType), bbox_inches='tight')
    plt.clf()



def preprocessPriods(dfGroup, columns, listPermu, dictSplit, folder,listDateRange, x_length, y_length, sampling_method, model_type):
    # We have to append instances of different periods as different dataframes because later
    # they would be transform to sequences
    
    
    dfGroup = dfGroup.reset_index()    
    groupType = dfGroup['Group Type'].loc[0]

    trainList, valList, testList = getSplitInPeriod(dfGroup, listPermu, dictSplit,listDateRange)
    oneSeriesTrain = pd.concat(trainList)
    oneSeriesVal = pd.concat(valList)
    oneSeriesTest = pd.concat(testList)

    #In case there are no train instances at all:
    if oneSeriesTrain.shape[0]==0:
        #hf.close()
        return  (None,None,None,None,None,None),(None,None,None,None),(None,None,None)

    #Get quantiles in training
    q50Train = oneSeriesTrain.quantile(0.5)['Response Time']
    q90Train = oneSeriesTrain.quantile(0.9)['Response Time']

    q50Valid = oneSeriesVal.quantile(0.5)['Response Time']
    q90Valid = oneSeriesVal.quantile(0.9)['Response Time']
    
    q50Test = oneSeriesTest.quantile(0.5)['Response Time']
    q90Test = oneSeriesTest.quantile(0.9)['Response Time']


    # Replace no close-date and outliers. Choices instead of sample to include repetition
    #  in cases where the sampled list is smaller, than the values to complete
    
    print(groupType, "qtrain50 and 90 ", q50Train, q90Train)

    nameFile = folder+groupType+'prev_feature_ones_grouped.csv'

    if path.exists(nameFile):

        dfGroup = pd.read_csv(nameFile)
        print("File read",nameFile)
        
    else:
        outliers, validPoints, dfGroup = replaceValues(oneSeriesTrain, dfGroup, q90Train, groupType, sampling_method)
        dfGroup.to_csv(nameFile)

    #print(dfGroup[columns])
    #None to make shape (x,1) needed for fit that expected 2D
    if model_type == None:
        train, validation, test, scaler = minMaxData(dfGroup['Response Time'].iloc[oneSeriesTrain.index][:,None],
                                                     dfGroup['Response Time'].iloc[oneSeriesVal.index][:,None], 
                                                     dfGroup['Response Time'].iloc[oneSeriesTest.index][:,None])
        """train, validation, test, scaler = minMaxData(np.array(dfGroup['Response Time'].iloc[oneSeriesTrain.index]).reshape(-1, 2),
                                                     np.array(dfGroup['Response Time'].iloc[oneSeriesVal.index]).reshape(-1, 2), 
                                                     np.array(dfGroup['Response Time'].iloc[oneSeriesTest.index]).reshape(-1, 2))"""

    else:
        train = dfGroup['Response Time'].iloc[oneSeriesTrain.index][:,None]
        validation = dfGroup['Response Time'].iloc[oneSeriesVal.index][:,None]
        test = dfGroup['Response Time'].iloc[oneSeriesTest.index][:,None]
        scaler = None
    
    #print("Here",np.array(dfGroup['Subtype'].iloc[oneSeriesTrain.index]))#.to_string())#,np.squeeze(train))

    #print(train)

    newDFGroup = pd.DataFrame()
    newDFGroup['Response Time'] = dfGroup['Response Time']
    if len(columns) > 1:
        newDFGroup['Subtype'] = dfGroup['Subtype']
    newDFGroup['Response Time'].iloc[oneSeriesTrain.index] = np.squeeze(train)
    newDFGroup['Response Time'].iloc[oneSeriesVal.index] = np.squeeze(validation)
    newDFGroup['Response Time'].iloc[oneSeriesTest.index] = np.squeeze(test)

    #print(newDFGroup)

    #Update quantiles including the values being replaced
    q50Train = newDFGroup['Response Time'].iloc[oneSeriesTrain.index].quantile(0.5)
    q90Train = newDFGroup['Response Time'].iloc[oneSeriesTrain.index].quantile(0.9)
    q50Valid = newDFGroup['Response Time'].iloc[oneSeriesVal.index].quantile(0.5)
    q90Valid = newDFGroup['Response Time'].iloc[oneSeriesVal.index].quantile(0.9)
    q50Test = newDFGroup['Response Time'].iloc[oneSeriesTest.index].quantile(0.5)
    q90Test = newDFGroup['Response Time'].iloc[oneSeriesTest.index].quantile(0.9)

    # Transform to sequences
    newDFTrainX , newDFTrainY, idxTrainX, idxTrainY = transformXY(trainList, newDFGroup, columns, x_length, y_length)
    newDFValX , newDFValY , idxValX, idxTValY= transformXY(valList, newDFGroup, columns, x_length, y_length)
    newDFTestX, newDFTestY, idxTestX, idxTestY = transformXY(testList, newDFGroup, columns, x_length, y_length)
            
    if len(newDFTrainX)==0 or len(newDFValX)==0 or len(newDFTestX)==0:
        #hf.close()
        return (None,None,None,None,None,None),(None,None,None,None),(None,None,None)

    newDFTrainX_new = []
    newDFTrainY_new = []
    idxTrainX_new = []
    idxTrainY_new = []
    for i in range(len(newDFTrainX)):
        for j in range(len(newDFTrainX[i])):
            #print(i,j,np.asarray(newDFTrainX[i][j]))
            newDFTrainX_new.append(np.asarray(newDFTrainX[i][j]))
            newDFTrainY_new.append(np.asarray(newDFTrainY[i][j]))
            idxTrainX_new.append(np.asarray(idxTrainX[i][j]))
            idxTrainY_new.append(np.asarray(idxTrainY[i][j]))


    newDFValX_new = []
    newDFValY_new = []
    idxValX_new = []
    idxValY_new = []
    for i in range(len(newDFValX)):
        for j in range(len(newDFValX[i])):
            #print(i,j,np.asarray(newDFTrainX[i][j]))
            newDFValX_new.append(np.asarray(newDFValX[i][j]))
            newDFValY_new.append(np.asarray(newDFValY[i][j]))
            idxValX_new.append(np.asarray(idxValX[i][j]))
            idxValY_new.append(np.asarray(idxTValY[i][j]))


    newDFTestX_new = []
    newDFTestY_new = []
    idxTestX_new = []
    idxTestY_new = []
    for i in range(len(newDFTestX)):
        for j in range(len(newDFTestX[i])):
            #print(i,j,np.asarray(newDFTrainX[i][j]))
            newDFTestX_new.append(np.asarray(newDFTestX[i][j]))
            newDFTestY_new.append(np.asarray(newDFTestY[i][j]))
            idxTestX_new.append(np.asarray(idxTestX[i][j]))
            idxTestY_new.append(np.asarray(idxTestY[i][j]))
    
    newDFTrainX_new = np.array(newDFTrainX_new)
    newDFTrainY_new = np.array(newDFTrainY_new)
    newDFValX_new = np.array(newDFValX_new)
    newDFValY_new = np.array(newDFValY_new)
    newDFTestX_new = np.array(newDFTestX_new)
    newDFTestY_new = np.array(newDFTestY_new)

    idxTrainX_new = np.array(idxTrainX_new)
    idxTrainY_new = np.array(idxTrainY_new)
    idxTestX_new = np.array(idxTestX_new)
    idxTestY_new = np.array(idxTestY_new)
    idxValX_new = np.array(idxValX_new)
    idxValY_new = np.array(idxValY_new)

    #print("TestX:",newDFTestX_new)
    #print("TestY:",newDFTestY_new[:,:,0,None])

    #print( newDFTrainX_new.shape, newDFValX_new.shape, newDFTestX_new.shape, newDFTrainY_new.shape, newDFValY_new.shape, newDFTestY_new.shape )
    #print( np.array(newDFTrainX_new).shape, np.array(newDFValX_new).shape, np.array(newDFTestX_new).shape )

    #Creating H5FD file
    '''
    hf.create_dataset('trainX', data=newDFTrainX)
    hf.create_dataset('trainY', data=newDFTrainY)
    hf.create_dataset('validX', data=newDFValX)
    hf.create_dataset('validY', data=newDFValY)
    hf.create_dataset('testX',  data=newDFTestX)
    hf.create_dataset('testY',  data=newDFTestY)
    hf.create_dataset('quantiles', data = [q50Train, q90Train, q50Valid, q90Valid, q50Test, q90Test])
    hf.create_dataset('scaler', data=scaler)
    
    hf.close()


    '''
    
    saveHistogramPlots(groupType, newDFTrainX_new, newDFValX_new,newDFTestX_new, newDFTrainY_new,  newDFValY_new, newDFTestY_new, folder)

    print( q50Train, q90Train, q50Valid, q90Valid, q50Test, q90Test)


    #print(newDFTrainX)
    return (newDFTrainX_new, newDFValX_new,newDFTestX_new, newDFTrainY_new[:,:,0,None],  newDFValY_new[:,:,0,None], newDFTestY_new[:,:,0,None]), \
            (scaler, q50Train, q90Train, q50Valid, q90Valid, q50Test, q90Test), \
            (dfGroup, idxTestX_new, idxTestY_new)



def getPreProcessedData(groupType, px_length, py_length, periods, sampling_method, columns = ['Response Time'], model_type=None):
    if periods==8:
        listDateRange = listDateRange8
    else:
        listDateRange = listDateRange17

    x_length = px_length
    y_length = py_length

    #print(x_length, y_length)
    folder , h5file = createFolder( listDateRange, groupType, sampling_method, periods)

    if h5file == None:
        pass
    else:
        return h5file


    df = pd.read_csv("./dataprocess/Emergency_Response_Incidents.csv", error_bad_lines=False, index_col=False) #9018
    df = createResponseTimeColumn(df)

    listDfPerGroup, df , listGroupsNames = splitGroupTypes(df)

    print("Number of Type Groups found: ", len(listDfPerGroup))

    #Set split of training percentages
    dictSplit = {key: None for key in listSplits} 
    dictSplit['train']		=	trainPercentage
    dictSplit['validation']	=	validPercentage
    dictSplit['test']		=	testPercentage

    # Order in which to take train, val and test from each period
    #listPermu = list(it.permutations(listSplits))

    if groupType==None:
    	for dfGroup in listDfPerGroup:
    		newDFTrainX, newDFValX,newDFTestX, newDFTrainY,  newDFValY, newDFTestY, scaler = preprocessPriods(dfGroup,columns,listPermu, dictSplit, folder,listDateRange, x_length, y_length, sampling_method,model_type)
    else:
        idxGroup = listGroupsNames.index(groupType)
        dfGroup = listDfPerGroup[idxGroup]
        sets, stats, indexRef=  preprocessPriods(dfGroup, columns, listPermu, dictSplit, folder,listDateRange, x_length, y_length, sampling_method,model_type)

        return sets, stats, indexRef
        #(newDFTrainX, newDFValX,newDFTestX, newDFTrainY,  newDFValY, newDFTestY), (scaler, q50Train, q90Train, q50Valid, q90Valid, q50Test, q90Test),() = preprocessPriods(dfGroup, listPermu, dictSplit, folder,listDateRange, x_length, y_length)
    	#return newDFTrainX, newDFValX,newDFTestX, newDFTrainY,  newDFValY, newDFTestY, scaler , q50Train, q90Train, q50Valid, q90Valid, q50Test, q90Test



#getPrProcessedData('Fir')


############# PENDING   #######################
### Save indexes of what was training, val and split form the origina
### Save SCALER

#Give back the scaler too
"""getPreProcessedData("Fir", 15, 5, 8, "Random")
getPreProcessedData("Law", 15, 5, 8, "Random")
getPreProcessedData("Str", 15, 5, 8, "Random")"""
#getPreProcessedData("Law", 15, 5, 8, "Dist", ['Response Time'],"lstm")