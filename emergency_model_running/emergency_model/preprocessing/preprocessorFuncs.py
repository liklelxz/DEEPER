import numpy as np
import pandas as pd
#Preprocessng
from sklearn.preprocessing import MinMaxScaler


#-----------------------------------------
# helper functions for getData()
#-----------------------------------------

def sampleData(dataset,x_length,y_length):
    x_data_limit = len(dataset) - (x_length+y_length)
    X = []
    Y = []
    for i in range(x_data_limit):
        # for the inputs
        temp_x = []
        for j in range(x_length):
            temp_x.append(dataset[i+j])
        X.append(temp_x)
        # for the outputs
        temp_y = []
        for j in range(y_length):
            temp_y.append(dataset[i+x_length+j])
        Y.append(temp_y)
    return X,Y
        

def split(all_data, percentage):
    train_data_size = int(percentage*len(all_data))#431
    validation_data_size = train_data_size + int(0.5*(1-percentage)*len(all_data))
    train_data = all_data[:train_data_size]
    valid_data = all_data[train_data_size : validation_data_size]
    test_data = all_data[validation_data_size:]
    print(len(train_data),len(valid_data), len(test_data))
    return train_data, valid_data, test_data


def minMaxData(train_data, valid_data, test_data):
    #Preprocessing
    scaler = MinMaxScaler()#feature_range=(-1, 1)
    scaler.fit(train_data.reshape(-1, 1))
    train_normalized = scaler.transform(train_data.reshape(-1, 1))
    valid_normalized = scaler.transform(valid_data.reshape(-1,1))
    test_normalized = scaler.transform(test_data.reshape(-1, 1))

    return train_normalized, valid_normalized, test_normalized, scaler


def preprocessing(file_name, percentage):
    #Reading and spliting
    df = pd.read_csv(file_name, delimiter=',')
    all_data = df['Response Time'].values.astype(float)#
    train_data, valid_data, test_data = split(all_data, percentage)
    train_normalized, valid_normalized, test_normalized, scaler = minMaxData(train_data, valid_data, test_data)

    #Calculate quantiles
    qTra50 = np.quantile(train_normalized,0.5)
    qTra90 = np.quantile(train_normalized,0.9)
    qVal50 = np.quantile(valid_normalized,0.5)
    qVal90 = np.quantile(valid_normalized,0.9)
    qTest50 = np.quantile(test_normalized,0.5)
    qTest90 = np.quantile(test_normalized,0.9)
    print("Q-0.5: ",qTra50, qVal50,qTest50 )
    print("Q-0.9: ",qTra90, qVal90,qTest90 )

    return train_normalized, valid_normalized, test_normalized, scaler, qTra50, qVal50,  qVal90 , qTest90


    
#-----------------------------------------
# main method to obtain data
#-----------------------------------------

# obtains the datasets -> used for the RNN model
# filename : the string name for the file
# x_length : length of the input(timesteps of the past)
# y_length : length of output(timesteps into future)
# percentage : the percentage of data to use for training and testing



def stackHorizontally(dataset,input_dim=1):
    newSet = []
    for i in range(0,len(dataset)):
        instance = np.hstack(dataset[i])
        newSet.append(instance)
    return np.array(newSet)

#75% - 25% in order
def getData(filename,x_length,y_length,percentage, input_dim=1, columns='R', validation=0, typeModel=0, limitOutlier=300000):

    df = pd.read_csv(filename, delimiter=',')#,usecols=["Water flow"]
    #print df.columns
    median = df.loc[df['Response Time']<=limitOutlier, 'Response Time'].median()
    df.loc[df['Response Time'] > limitOutlier, 'Response Time'] = np.nan
    df.fillna(median,inplace=True)

    #Selecting columns for the dataset
    listCols = {'R':"Response Time"}
    listCols = [ listCols[column] for column in columns]

    #print listCols
    data = df[listCols]

    #-- seperate training and testing --------
    train_size = int(percentage*len(data))    
    train_data = np.array(data[:train_size])

    if validation == 0:
        test_data = np.array(data[train_size:])

        qTest90=np.quantile(test_data,0.9)

        X_Train,Y_Train = sampleData(train_data,x_length,y_length)
        X_Test,Y_Test = sampleData(test_data,x_length,y_length)

        X_Train,Y_Train,X_Test,Y_Test = np.array(X_Train),np.array(Y_Train)[:,:,0,None],np.array(X_Test),np.array(Y_Test)[:,:,0,None]

        print (len(data), train_size, X_Train.shape, Y_Train.shape, X_Test.shape, Y_Test.shape, qTest90)
        return X_Train,Y_Train,X_Test,Y_Test, qTest90
    else:
        validation_data_size = train_size + int(0.5*(1-percentage)*len(data))
        test_data = np.array(data[validation_data_size:])

        qTest90=np.quantile(test_data,0.9)

        X_Train,Y_Train = sampleData(train_data,x_length,y_length)
        X_Test,Y_Test = sampleData(test_data,x_length,y_length)

        X_Train,Y_Train,X_Test,Y_Test = np.array(X_Train),np.array(Y_Train)[:,:,0,None],np.array(X_Test),np.array(Y_Test)[:,:,0,None]

        print (len(data), train_size, len(test_data), X_Train.shape, Y_Train.shape, X_Test.shape, Y_Test.shape, qTest90)
        return X_Train,Y_Train,X_Test,Y_Test, qTest90

#getData('./../Data/FO.csv',10,5,0.75,input_dim=1, columns='R', typeModel=0,limitOutlier=30000)
