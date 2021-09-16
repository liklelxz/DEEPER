import pandas as pd
import argparse
from datetime import datetime,date
import matplotlib.pyplot as plt 

# 
save_path = '../Data/'

def split(resource):

	df = pd.read_csv(save_path+resource+'.csv', header=0) # , squeeze=True, index_col=0,
	
	df1 = df["Incident Type"].value_counts().reset_index(name="count")

	#remove incidents that occur 5 times or less
	indexNames = df1[ df1["count"] > 6 ].index
	df1.drop(indexNames , inplace=True)

	cond = df["Incident Type"].isin(df1['index'])
	df.drop(df[cond].index, inplace = True)

	df.to_csv(save_path+resource+'_remove_small.csv')


parser = argparse.ArgumentParser(description='lstm')
parser.add_argument('--resource', type=str, default='',
                    help='Enter the source for which it is running, W: water or E: energy')

args = parser.parse_args()
split(args.resource)

#Run script as follows: python modify_dataset.py --resource FO