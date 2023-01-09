import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pylab as pl
import argparse

def parameter_parser():
    '''
    for parsing args (cmd)
    '''
    parser = argparse.ArgumentParser(description="Run IoTSensorClustering.")

    parser.add_argument('-datapath',
                        nargs='?',
                        default='SensorsDataSet/28_1800.csv',
                        help='input dataset (.csv) .')

    parser.add_argument('-k',
                        nargs='?',
                        default=4,
                        help='input k (num of cluster) .')

    return parser.parse_args()


def ReadData(path):
    '''
    param. path(str)
    return. table(dataframe.pivot_table)
    '''
    df = pd.read_csv(path)
    table = df.pivot_table(index=["Name"])
    table = table.reset_index()
    return table


def Preprocessing(table,feature=["AirTemp", "RelativeHumidity", "WindSpeed"]):
    '''
    param. 
        table(dataframe.pivot_table)
        feature(list) : select the feature we wanna use in clustering
    return.
        table(dataframe.pivot_table)
    description.
        select the column we want
        drop the rows which contain missing value
        normalize
    '''
    # for index, x, y
    feature_list = ['Name', 'Lat', 'Longt']
    feature_list.extend(feature)

    # select the feature & dropna
    table2 = table[feature_list]
    table2 = table2.dropna()
    # normalize
    scaler = MinMaxScaler()
    table2.iloc[:, 3:] = scaler.fit_transform(table2.iloc[:, 3:])

    return table2


def Clustering(k,data):
    '''
    param.
        k : num of clustering
        data
    return.
        data
    description.
        KMeans
        save data (contain index, x, y, FEATURE, Cluster)
    '''
    # clustering
    cluster = KMeans(n_clusters=k, random_state=0, init='random')
    data["Cluster"] = cluster.fit_predict(data[data.columns[3:]])
    data.to_csv('ClusteredData_modified.csv', index=False)
    return data

def Plot(weather_clusters,feature=["AirTemp", "RelativeHumidity", "WindSpeed"]):
    '''
    param. 
        weather_clusters
        feature
    No return.
    save result(.png)
    '''
    # plotting clusters
    plt.figure(num=None, figsize=(16, 9), dpi=80)

    x = dict()
    y = dict()
    c = dict()
    color = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    fullColor = {
        'r':'Red'
        ,'g':'Green'
        ,'b':'Blue'
        ,'y':'Yellow'
        ,'c':'Cyan'
        ,'m':'Magenta'
        ,'k':'Black'
    }

    k = len(weather_clusters.Cluster.value_counts())
    for i in range(k):
        x[i] = weather_clusters[weather_clusters.Cluster == i]['Longt']
        y[i] = weather_clusters[weather_clusters.Cluster == i]['Lat']
        c[i] = pl.scatter(x[i], y[i], c=color[i], marker='o', alpha=0.4)
        print('Cluster', i, ',Size:', '%5d'%len(x[i]), ', Color: ',fullColor[color[i]])


    pl.xlabel('Longitude')
    pl.ylabel('Latitude')
    pl.title(', '.join(weather_clusters.columns[3:-1]))
    pl.savefig("plot_output_modified.png")
    pl.show()

if __name__=='__main__':
    args = parameter_parser()
    print(args.datapath)
    data = ReadData(args.datapath)

    feature=["AirTemp", "RelativeHumidity", "WindSpeed"]

    data = Preprocessing(data,feature)
    data = Clustering(int(args.k), data)
    Plot(data)