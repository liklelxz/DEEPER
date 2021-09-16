import pandas as pd
import numpy as np
import datetime
import argparse

parser = argparse.ArgumentParser(description='probability_counter')
parser.add_argument('--inputpath', type=str, default='',
                    help='where the input csv located')
parser.add_argument('--outputpath', type=str, default='',
                    help='where the output csv located')
parser.add_argument('--dayorweek', type=int,default=1,
                    help='judge probabilities by day or week')
args = parser.parse_args()
inputpath = args.inputpath
outputpath = args.outputpath
dow = args.dayorweek

def pro_filter(inputpath, outputpath, dow = 1):
    df= pd.read_csv(inputpath,delimiter=",",index_col=None)
    df["pro_col"] = df["Incident Type"].str.split(pat = "-").str.get(0)
    #print df["pro_col"].value_counts()
    df["Creation Date"] = pd.to_datetime(df["Creation Date"])
    df = df.sort_values(by="Creation Date")
    df['Week'] = pd.to_datetime(df["Creation Date"]).dt.strftime('%Y-%U')
    df['day'] = pd.to_datetime(df["Creation Date"]).dt.strftime('%D')
    #df1 = pd.DataFrame(df.pro_col.value_counts().reset_index())
    #df1 =  df.groupby(["pro_col"]).size()
    #df1.to_csv(outputpath)
    if (dow):
        week_pro(df,outputpath)
    else:
        day_pro(df,outputpath)






def week_pro(df,outpath):
    # df4 = df.groupby(["pro_col"],as_index=False)["Week"]
    df2 = df.groupby("pro_col")["Week"]
    l1 = list(df2)
    # print l1

    df3 = pd.DataFrame()
    # df3['Week'] = pd.date_range('5/4/2011', '12/17/2019', freq='W')
    # df3['Week'] = df3["Week"].dt.strftime('%Y-%U')
    # df3["Week"] = pd.Series(df["Week"])
    # df3[l1[2][0]] = l1[2][1]
    # print type(l1[2][1])
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df3["Week"] = df["Week"]

    df5["Week1"] = pd.Series(pd.date_range('5/1/2011', '12/17/2019', freq='W'))

    df5["Week"] = pd.to_datetime(df5["Week1"]).dt.strftime('%Y-%U')
    df4["Week"] = df5["Week"]
    df4 = df4.set_index(np.array(df4["Week"]))

    for x in range(0, len(l1)):
        df3[l1[x][0]] = l1[x][1]
        g1 = df3[l1[x][0]].value_counts().to_frame()
        g1 = g1.rename(columns={l1[x][0]: ("count" + str(x))})
        g1[l1[x][0]] = g1.index
        # g1 = g1.sort_values(by=l1[x][0])
        df4[l1[x][0]] = pd.Series(g1[l1[x][0]])
        df4[("count" + str(x))] = pd.Series(g1[("count" + str(x))])
    df4["Sum"] = 0
    for t in range(0, len(l1)):
        df4[("count" + str(t))] = df4[("count" + str(t))].fillna(0)
        df4["Sum"] += df4[("count" + str(t))]

    for x in range(0, len(l1)):
        df4[l1[x][0]] = df4["count" + str(x)] / df4["Sum"]
    df6 = pd.DataFrame()
    df6["Week"] = df4["Week"]
    for x in range(0, len(l1)):
        df6[l1[x][0] + "-"] = df4[l1[x][0]]
    df6["Sum"] = df4["Sum"]
    # g1 = (df_fire.groupby("Fire",as_index=False)["Fire"]).value_counts().reset_index(name="count")
    # g1 = g1.sort_values(by=l1[2][0])

    # df3.to_csv("test2.csv",index=False)

    df6.to_csv(outpath, index=False)
    print('success!')



def day_pro(df,outpath):
    df2 = df.groupby("pro_col")["day"]
    l1 = list(df2)
    #print l1

    df3 = pd.DataFrame()
    # df3['Week'] = pd.date_range('5/4/2011', '12/17/2019', freq='W')
    # df3['Week'] = df3["Week"].dt.strftime('%Y-%U')
    # df3["Week"] = pd.Series(df["Week"])
    # df3[l1[2][0]] = l1[2][1]
    # print type(l1[2][1])
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df3["Day"] = df["day"]

    df5["Day"] = pd.Series(pd.date_range('5/4/2011', '12/17/2019', freq='D'))

    df5["Day"] = pd.to_datetime(df5["Day"]).dt.strftime('%D')
    df4["Day"] = df5["Day"]
    df4 = df4.set_index(np.array(df4["Day"]))

    for x in range(0, len(l1)):
        df3[l1[x][0]] = l1[x][1]
        g1 = df3[l1[x][0]].value_counts().to_frame()
        g1 = g1.rename(columns={l1[x][0]: ("count" + str(x))})
        g1[l1[x][0]] = g1.index
        #print g1
        # g1 = g1.sort_values(by=l1[x][0])
        df4[l1[x][0]] = pd.Series(g1[l1[x][0]])
        df4[("count" + str(x))] = pd.Series(g1[("count" + str(x))])
    df4["Sum"] = 0
    for t in range(0, len(l1)):
        df4[("count" + str(t))] = df4[("count" + str(t))].fillna(0)
        df4["Sum"] += df4[("count" + str(t))]

    for x in range(0, len(l1)):
        df4[l1[x][0]] = df4["count" + str(x)] / df4["Sum"]
    df6 = pd.DataFrame()
    df6["Day"] = df4["Day"]
    for x in range(0, len(l1)):
        df6[l1[x][0] + "-"] = df4[l1[x][0]]
    df6["Sum"] = df4["Sum"]
    df6 = df6.fillna(0)
    # g1 = (df_fire.groupby("Fire",as_index=False)["Fire"]).value_counts().reset_index(name="count")
    # g1 = g1.sort_values(by=l1[2][0])

    # df3.to_csv("test2.csv",index=False)

    df6.to_csv(outpath, index=False)
    print('success!')
	

pro_filter(inputpath,outputpath,dow)
