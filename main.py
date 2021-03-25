import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ijson
import json
import sys

from pandas.io.json import json_normalize
sns.set()



import re
import sys
import json
import re, string


import random
from tqdm import tqdm

from Kmeans import kMeans


def loss_kmeans(tweets,k,df_json):
	
	
    
    	ids=list(df_json['id'])
    	random.shuffle(ids)
    	seeds=ids[0:k]
    	kmeans = kMeans(seeds, tweets)
    	kmeans.converge()
    	result=kmeans.clusters
    
    	_,_,square_dist=kmeans.calcNewClusters()

    
    	return square_dist
    
if __name__=="__main__":

	filename=sys.argv[1]

	with open(filename,'r') as file:
		
	    	data = json.load(file)
	    
	df_json = pd.DataFrame(data)

	tweets={}
	for i in range(len(df_json['id'])):
		
		
	    	temp = df_json['text'][i]
	    	temp = temp.replace("'","")
	    	temp = temp.replace("[","")
	    	temp = temp.replace("]","")
	    	df_json['text'][i]=temp
	    
	    
	    	tweets[df_json['id'][i]] = temp


	    
	    
	loss=[]
	cluster_size=[]
	for i in tqdm(range(3,30)):
		
	    	loss.append(loss_kmeans(tweets=tweets,k=i,df_json=df_json))
	    	cluster_size.append(i+1)
	    	
	plt.plot(cluster_size,loss)
	plt.xlabel('Number of clusters')
	plt.ylabel('Loss')

	val = input("Enter your value: ")

	k=int(val)
	ids=list(df_json['id'])
	random.shuffle(ids)
	seeds=ids[0:k]
	kmeans = kMeans(seeds, tweets)
	kmeans.converge()
	result=kmeans.clusters
	
	
	col=[]
	numbers=[]
	for i in result:
	
		print("Cluster:  -->",i)
		print('-'*100)
		col.append("cluster "+str(i))
		cnt=0
	    	
		for j in result[i]:
	  
			cnt+=1
			print(j)
			print(tweets[j])
	    
		numbers.append(cnt)
	    
	
	
	df=pd.DataFrame([],columns=col)
	df_length = len(df)
	df.loc[df_length] = numbers
	
	
	fig = plt.figure(figsize =(10, 7)) 
	#a = new_groups_df.drop(['total'], axis = 1)
	plt.pie(df.loc[df_length], labels = df.columns)
	plt.title('A pie chart showing the volumes of tweets under different categories.')
	plt.show()



