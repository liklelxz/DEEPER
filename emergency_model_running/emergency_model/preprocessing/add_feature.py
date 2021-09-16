import pandas as pd
import numpy as np
import argparse
from datetime import datetime,date
import matplotlib.pyplot as plt 

# Change the save path for respective distribution type
#save_path = 'Data/MixedPeriods/2011-05_2012-05_8_Random/'
save_path = 'Data/MixedPeriods/2011-05_2012-05_8_Dist/'

def split(resource):

	df = pd.read_csv(save_path+resource+'.csv', header=0) # , squeeze=True, index_col=0,

	subtypes = df["Incident Type"].unique()
	print(subtypes)
	
	df1 = df["Incident Type"].value_counts()#.reset_index(name="count")
	print(df1)
	one_subtypes = df1[df1 <= 1].index

	filtered = df[df["Incident Type"].isin(one_subtypes)]

	one_subtypes = filtered["Incident Type"].unique()

	subtypes = np.array([x for x in subtypes if x not in one_subtypes])

	print("\n One subtypes: ",one_subtypes)
	print("\n Non one subtypes: ",subtypes)

	df_subtype = []

	for index, row in df.iterrows():
		#subtypes.index(df["Incident Type"][index])
		if df["Incident Type"].iloc[index] in one_subtypes:
			print("Yes",df["Incident Type"].iloc[index])
			df_subtype.append(0)
		else:
			value = np.where(subtypes == (df["Incident Type"].iloc[index]))[0][0]
			df_subtype.append(value+1)
		#print(df["Incident Type"].iloc[index],np.where(subtypes == (df["Incident Type"].iloc[index]))[0][0]+1)

	df["Subtype"] = df_subtype
	print(df)
	
	df.to_csv(save_path+resource+"_feature_ones_grouped"+".csv", index=False)
	
parser = argparse.ArgumentParser(description='lstm')
parser.add_argument('--resource', type=str, default='',
                    help='Enter the source for which it is running, W: water or E: energy')

args = parser.parse_args()
split(args.resource)

#Run script as follows: python add_feature.py --resource Fireprev