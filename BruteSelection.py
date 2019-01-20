#project specific functions
from functions import dataframe_splitter, build_models, build_new_polynomial_frame

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import itertools as it
from multiprocessing import Pool #Process
import os



#put file paths in these strings
inputcsv = ''
featureoutput = ''
polynomialoutput = '' #so you can view the additional generated features if desired
blueprintoutput = ''


#Parameters
label = 'quality'
use_polynomial = False
degree = 2 #determines x^degree of the PolynomialFeature
model = LinearRegression()

#for distributing calculations onto multiple CPU cores -> speed up compuations
cores = os.cpu_count()


#make sure the label provided is actually in the df columns
df = pd.read_csv(inputcsv)

if label not in df.columns.tolist():
    print('That\'s not a valid label')
    quit()

if use_polynomial:
	df = build_new_polynomial_frame(data=df,label=label,degree=degree)

columns = df.columns.tolist()
columns.remove(label)


#Get the number of columns
length = len(columns)
#figure out how many possible combinations there are. There may be a way to rewrite this and make this loop less redundant.
total = 0
for x in range(1,length+1):
    total += len(list(it.combinations(columns,x)))
print(total,' possible unique feature sets')

#create a datframe to house all the possible feature combinations.
#This DataFrame will be looped through later. This acts as a blue print for what models to build and test.
blue_print = pd.DataFrame(data=None,index=range(0,total),columns=['Feature List','Number of Features','R Squared','Coefficients'])

#throw those feature combos into the dataframe for later computation.
#a little redundant. possibly streamline later.
index = 0
for z in range(1,length+1):
    for x in list(it.combinations(columns,z)):
        blue_print.at[index,'Feature List'] = list(x)
        blue_print.at[index,'Number of Features'] = len(list(x))
        index += 1

#for debugging purposes
#blue_print.to_csv(blueprintoutput,index=False)

#takes the blue print and breaks it up into separate chunks to process in parallel on separate CPU's later down
splits = dataframe_splitter(dataframe=blue_print,sections=cores)


#for 1 CPU core
results = build_models(data=df,label=label,estimator=model,blue_print=blue_print)

results.sort_values('R Squared',inplace=True,ascending=False)
results.reset_index(drop=True,inplace=True)

results.to_csv(featureoutput,index=False)