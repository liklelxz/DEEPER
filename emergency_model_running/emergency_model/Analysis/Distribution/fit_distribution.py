import warnings
import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

listDateRange17 = [('2011-05','2011-11'),('2011-11','2012-05'),('2012-05','2012-11'),('2012-11','2013-05'),
                 ('2013-05','2013-11'),('2013-11','2014-05'),('2014-05','2014-11'),('2014-11','2015-05'),
                 ('2015-05','2015-11'),('2015-11','2016-05'),('2016-05','2016-11'),('2016-11','2017-05'),
                 ('2017-05','2017-11'),('2017-11','2018-05'),('2018-05','2018-11'),('2018-11','2019-05'),
                 ('2019-05','2020-01')]

listDateRange8 = [('2011-05','2012-05'),('2012-05','2013-05'),('2013-05','2014-05'),('2014-05','2015-05'),
                 ('2015-05','2016-05'),('2016-05','2017-05'),('2017-05','2018-05'),('2018-05','2020-01')]

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
save_path = './Analysis/Distribution/'

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

def replaceValues(validPointsTrain, dfGroup, q90Train, groupType):

    print('Outliers: ', dfGroup.loc[(dfGroup['Response Time']>q90Train)].shape[0], 'Null: ',  dfGroup.loc[(dfGroup["Response Time"].isna()==True)].shape[0] )

    outliers = dfGroup.loc[(dfGroup['Response Time']>q90Train) | (dfGroup["Response Time"].isna()==True)]
    validPoints = validPointsTrain.loc[(validPointsTrain['Response Time']<=q90Train)]

    print(validPoints,validPoints.shape,outliers.shape)

    """distributions = [
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]"""

    # Divide the observed data into 100 bins for plotting (this can be changed)
    y = validPoints['Response Time']
    x = np.arange(len(y))
    number_of_bins = 100
    bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99),number_of_bins)

    # Create the plot
    h = plt.hist(y, bins = bin_cutoffs, color='0.75')

    # Get the top three distributions from the previous phase
    dist_names = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','cauchy','chi','chi2','cosine',
        'dgamma','dweibull','erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk',
        'foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','genpareto','gennorm','genexpon',
        'genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r',
        'gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss',
        'invweibull','johnsonsb','johnsonsu','ksone','kstwobign','laplace','levy','levy_l',
        'logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf',
        'nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal',
        'rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda',
        'uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max','wrapcauchy'
    ]

    #dist_names = ['fatiguelife','mielke','nct']

    # Create an empty list to stroe fitted distribution parameters
    parameters = []
    p_values = []

    # Loop through the distributions ot get line fit and paraemters
    for dist_name in dist_names:
        # Set up distribution and store distribution paraemters
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        parameters.append(param)

        p = np.around(st.kstest(y,dist_name,args=param)[1],5)
        p_values.append(p)
        print(dist_name, param, p)
        
        # Get line for each distribution (and scale to match observed data)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
        pdf_fitted *= scale_pdf

        # Add the line to the plot
        plt.plot(pdf_fitted, label=dist_name)
        
        # Set the plot x axis to contain 99% of the data
        # This can be removed, but sometimes outlier data makes the plot less clear
        plt.xlim(0,np.percentile(y,99))
        plt.ylim(0,y.max())

    # Add legend and display plot

    best_index = p_values.index(max(p_values))
    best_fit = dist_names[best_index]
    best_fit_param = parameters[best_index]
    print("Best distribution:", best_index, best_fit, max(p_values))

    plt.legend()
    #plt.show()

    plt.savefig("{}_all_fits".format(save_path+groupType), bbox_inches='tight')
    plt.clf()

    # Store distribution paraemters in a dataframe (this could also be saved)
    dist_parameters = pd.DataFrame()
    dist_parameters['Distribution'] = (dist_names)
    dist_parameters['Distribution parameters'] = parameters
    dist_parameters['P_Values'] = p_values

    # Print parameter results
    """print ('\nDistribution parameters:')
    print ('------------------------')
    print (dist_parameters.to_string())"""

    dist_parameters.to_csv(save_path+groupType+'.csv')

    return best_fit, best_fit_param, p_values[best_index]


def preprocessPriods(dfGroup, listPermu, dictSplit, listDateRange):
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
        return None

    #Get quantiles in training
    q50Train = oneSeriesTrain.quantile(0.5)['Response Time']
    q90Train = oneSeriesTrain.quantile(0.9)['Response Time']

    q50Valid = oneSeriesVal.quantile(0.5)['Response Time']
    q90Valid = oneSeriesVal.quantile(0.9)['Response Time']
    
    q50Test = oneSeriesTest.quantile(0.5)['Response Time']
    q90Test = oneSeriesTest.quantile(0.9)['Response Time']

    print(groupType, "qtrain", q90Train)
    best_fit, best_fit_param, ks_test_pvalue = replaceValues(oneSeriesTrain, dfGroup, q90Train, groupType)
    return best_fit, best_fit_param, ks_test_pvalue

def getPreProcessedData(groupType, periods=8, model_type=None):
    if periods==8:
        listDateRange = listDateRange8
    else:
        listDateRange = listDateRange17
        
    df = pd.read_csv("./dataprocess/Emergency_Response_Incidents.csv", error_bad_lines=False, index_col=False) #9018
    df = createResponseTimeColumn(df)

    listDfPerGroup, df , listGroupsNames = splitGroupTypes(df)

    print("Number of Type Groups found: ", len(listDfPerGroup))

    #Set split of training percentages
    dictSplit = {key: None for key in listSplits} 
    dictSplit['train']      =   trainPercentage
    dictSplit['validation'] =   validPercentage
    dictSplit['test']       =   testPercentage

    if groupType==None:
        for dfGroup in listDfPerGroup:
            best_dist, best_fit_param, ks_test_pvalue = preprocessPriods(dfGroup, listPermu, dictSplit, listDateRange)
    else:
        idxGroup = listGroupsNames.index(groupType)
        dfGroup = listDfPerGroup[idxGroup]
        best_dist, best_fit_param, ks_test_pvalue = preprocessPriods(dfGroup, listPermu, dictSplit, listDateRange)

    print("Best fit {}: {}".format(groupType,best_dist))

    f = open(save_path+"best_distributions_all_types","a")
    f.write(groupType+'\t'+best_dist +'\t'+ str(best_fit_param)+'\t'+str(ks_test_pvalue)+'\n')
    f.close()
    return best_dist

getPreProcessedData("Fir")
getPreProcessedData("Law")
getPreProcessedData("Str")
getPreProcessedData("Uti")