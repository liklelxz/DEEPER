import pandas as pd
import numpy as np
import datetime
import argparse

parser = argparse.ArgumentParser(description='filter_csv')
parser.add_argument('--filepath', type=str, default='',
                    help='Path where the files are located')

parser.add_argument('--outpath', type=str, default='',
                    help='Path where the filtered csv located')
parser.add_argument('--outpath_count', type=str, default='',
                    help='Path where the count_styled filtered csv located')
parser.add_argument('--outpath_median', type=str, default='',
                    help='Path where the median filtered csv located')
parser.add_argument('--replace_median', type=int, default=1,
                    help='Flag to choose whether replace with median or maxrt')
parser.add_argument('--column',type=str,default='TCRYWD',help="column we want to get, must include TCR")
parser.add_argument('--classify',type=str,default='Incident Type',
                    help="Which column you want to use to classify")
#parser.add_argument('--type_filter',type=str,default='Fire',help="The incident type you want to filter")
parser.add_argument('--begindate',type=str,default="2011/05/04",
                    help="The begin date you want to filter, earliest date is 2011/05/04")
parser.add_argument('--enddate',type=str,default=None,
                    help="The end date you want to filter, latest date is 2019/12/17")
parser.add_argument('--Minrt',type=int, default= None,
                    help="Minimum response time")
parser.add_argument('--Maxrt',type=int,default=4000,
                    help="Maximum response time")
parser.add_argument('--Minevent',type=int,default=None,
                    help="Minimum number of same event")
parser.add_argument('--Maxevent',type=int,default=1000,
                    help="Maximum number of same event")
parser.add_argument('--combination_type',type=str,default=None,
                    help="Combine with another type")
args = parser.parse_args()

filepath = args.filepath
outpath = args.outpath
outpath_count = args.outpath_count
outpath_median = args.outpath_median
replace_median = args.replace_median
cl = args.column
clf = args.classify
bd = args.begindate
ed = args.enddate
minr = args.Minrt
maxr = args.Maxrt
mine = args.Minevent
maxe = args.Maxevent
combine = args.combination_type




def csvfilter1(filepath, outpath, outpath_median,replace_median = True, columns='TCRYWD', clasify="Incident Type", begindate="2011/05/04", enddate=None,
               Minrt=None, Maxrt=4000, Minevent=None,Maxevent=1000,combination_1= None):
    df = pd.read_csv(filepath, delimiter=",",index_col=None)
    if (combination_1 != "combination" and combination_1 != None):
        df = df[df[clasify].str.startswith(combination_1)]
    elif(combination_1 =="combination"):
        df = df[(df[clasify].str.startswith("Fire") == True)|(df[clasify].str.startswith("Utility") == True)|(df[clasify].str.startswith("Law Enforcement") == True)|(df[clasify].str.startswith("Structural") == True)]
    	
    
    df["Response Time"] = pd.Series()
    df1 = df[df["Closed Date"].isnull()]
    
    df1.to_csv("../Data/nan.csv",index=False)
    df = df.dropna(subset=["Closed Date"])
    df["Creation Date"]=pd.to_datetime(df["Creation Date"])
    df = df.sort_values(by=["Creation Date"])

    df["Response Time"] = (pd.to_datetime(df["Closed Date"]) - pd.to_datetime(df["Creation Date"])).dt.total_seconds()
    df["minute"] = 60
    df["Response Time"] = df["Response Time"].div(df["minute"])
    df["Response Time"] = df["Response Time"].astype(int)
    df["Creation Date"] = pd.to_datetime(df["Creation Date"])
    df["Creation Date"] = df["Creation Date"].dt.date
    df1["Creation Date"] = pd.to_datetime(df1["Creation Date"])
    df1["Creation Date"] = df1["Creation Date"].dt.date


    df["Creation Date"] = pd.to_datetime(df["Creation Date"])
    df1["Creation Date"] = pd.to_datetime(df1["Creation Date"])
    bd = pd.to_datetime(begindate)
    ed = pd.to_datetime(enddate)

    
    df = df[(df["Creation Date"] >= bd)]
    df1 = df1[(df1["Creation Date"] >= bd)]
    if (ed != None):
    	df = df[(df["Creation Date"] <= ed)]
    	df1 = df1[(df1["Creation Date"] <= ed)]
    
    if (Minrt != None):
    	df = df[(df["Response Time"] >= Minrt)]
    if(Maxrt!=None):
    	df["Response Time"].loc[df["Response Time"] >=Maxrt]=Maxrt
    #use loc to set conditon
    
    i = 0
   
    if (Minevent != None):
    	v = df[["Incident Type"]]
    	df=df[v.replace(v.apply(pd.Series.value_counts)).ge(Minevent).all(1)]
    	v1 = df1[["Incident Type"]]
    	df1=df1[v1.replace(v1.apply(pd.Series.value_counts)).ge(Minevent).all(1)]
    v2 = df[["Incident Type"]]
    df = df[v2.replace(v2.apply(pd.Series.value_counts)).le(Maxevent).all(1)]
    v3 = df1[["Incident Type"]]
    df1 = df1[v3.replace(v3.apply(pd.Series.value_counts)).le(Maxevent).all(1)]
    
    
    df["Year"] = pd.to_datetime(df["Creation Date"]).dt.year
    df['Week'] = df["Creation Date"].dt.strftime('%Y-%U')
    df["Day"] = pd.to_datetime(df["Creation Date"]).dt.dayofyear

    incident_type = df["Incident Type"].unique().tolist()
    

    df["Incident Number"] = [(incident_type.index(row)+1) for row in df["Incident Type"]]

    listCols = {'C': "Creation Date", 'R': "Response Time", 'B': "Borough", 'L': "Location", 'T': "Incident Type",'E': "Closed Date", 'N':"Incident Number", 'Y':"Year", 'W':"Week", 'D':"Day"}

    listCols = [listCols[column] for column in columns]
    df = df[listCols]
    df2 = df[["Incident Type","Response Time"]]
    df2 = df2.groupby(["Incident Type"],as_index=False).median()
    df2.to_csv(outpath_median,index=False)
    
    if(replace_median): 	
		for (index,row) in df1.iterrows():
		    for (index1,row1) in df2.iterrows():
		        if row.loc["Incident Type"] == row1.loc["Incident Type"]:
		    	    df1.loc[index,"Response Time"] = row1.loc["Response Time"]
		
	   
		df1["Year"] = pd.to_datetime(df1["Creation Date"]).dt.year
		df1['Week'] = pd.to_datetime(df1["Creation Date"]).dt.strftime('%Y-%U')
		df1["Day"] = pd.to_datetime(df1["Creation Date"]).dt.dayofyear
		df1 = df1[listCols]
		frames = [df,df1]
		df_join = pd.concat(frames)
    
		df_join["Creation Date"]=pd.to_datetime(df_join["Creation Date"])
		df_join = df_join.sort_values(by=["Creation Date"])
		df3 = df_join[df_join["Response Time"].isnull()]
		df_join = df_join.dropna(subset=["Response Time"])
		df_join.to_csv(outpath,index=False)
		df3.to_csv("../Data/no_closed.csv",index=False)
    else:
        df1["Year"] = pd.to_datetime(df1["Creation Date"]).dt.year
        df1['Week'] = pd.to_datetime(df1["Creation Date"]).dt.strftime('%Y-%U')
        df1["Day"] = pd.to_datetime(df1["Creation Date"]).dt.dayofyear
        df1 = df1[listCols]
        df1["Response Time"] = Maxrt
    	frames = [df,df1]
        df_join = pd.concat(frames)
    
        df_join["Creation Date"]=pd.to_datetime(df_join["Creation Date"])
        df_join = df_join.sort_values(by=["Creation Date"])
        df_join.to_csv(outpath,index=False)
		
	
	
    print("Success!")

   
       
def cal_count_week(inputpath, outputfile):
	#bdy = pd.to_datetime(begindate).dt.year
	#edy = pd.to_datetime(enddate).dt.year
	df = pd.read_csv(inputpath,delimiter=",",index_col=None)
	df1 = (df["Week"].value_counts()).to_frame();
	df1 = df1.rename(columns={"Week":"Events Count"})
	df1["Week"] = df1.index
	df1 = df1.sort_values(by=["Week"])
	df1.to_csv(outputfile, index=False)
	

	
csvfilter1(filepath,outpath,outpath_median,replace_median,cl,clf,bd,ed,minr,maxr,mine,maxe,combine)
cal_count_week(outpath,outpath_count)
#get_median(outpath,outpath_median)
#csvfilter1("~/Desktop/rt/Emergency_Response_Incidents.csv","~/Desktop/rt/FO_new.csv",columns="TCR",clasify='Incident Type',begindate="2014/03/01",enddate="2018/07/07",addweek=True)
