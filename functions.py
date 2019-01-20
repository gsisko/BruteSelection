import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import os


def dataframe_splitter(dataframe,sections):
	'''
	Give it a dataframe and tell it how many sections you want.
	It splits the dataframe based on the section count.
	Returns results as a list.
	'''

	length = len(dataframe)

	#iterate through the df with these - as the caterpillar inches along the leaf.
	lower = 0
	increment = length // sections
	upper = increment

	stragglers = length % sections #sometimes called the "remainder"
	datasets = [] #because this is dynamically instantiating objects, store in list, and use indeces to reference objects

	for x in range(sections-1):
		datasets.append(dataframe[lower:upper].copy())

		lower += increment
		upper += increment

	'''
	just tack on the remainder onto the last DataFrame.
	probabaly a better way to do this with math given that the remainder
	will never be greater than the divisor. This works for now.
	'''
	datasets.append(dataframe[lower:upper+stragglers])

	return datasets


def build_models(data,label,estimator,blue_print):
	'''
	Given a set of predetermined features combinations, build a model with every feature combination.
	Start iterating through the possible features.
	'''

	#First need to grab the length of the blue_print table.
	iterate = len(blue_print)
	for i in range(0,iterate):
		columns = blue_print.at[i,'Feature List']

		x = data[columns]
		y = data[label]
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3) #possibly going to move this outside the function later
		estimator.fit(x_train,y_train)
		y_predict = estimator.predict(x_test)

		blue_print.at[i,'R Squared'] = r2_score(y_test,y_predict)
		blue_print.at[i,'Coefficients'] = estimator.coef_

    	#print(i+1," out of ",total," calculated", end='\r')

	return blue_print


def build_new_polynomial_frame(data,label,degree):
	'''
	Acts as an extension of scikitlearn's PolynomialFeatures.
	Just takes a dataframe with the label, and creates polynomials
	for the features, and then joins the label back on.
	The PolynomialFeature Class transforms the label when no needed.
	'''
	columns = data.columns.tolist()
	columns.remove(label)

	features = data[columns]
	labelY = data[[label]]

	poly = PolynomialFeatures(degree)
	polynomials = poly.fit_transform(features)
	newnames = poly.get_feature_names(features.columns)

	dfp = pd.DataFrame(data=polynomials,columns=newnames)
	dfp = dfp.merge(labelY,how='inner',left_index=True,right_index=True)
	dfp = dfp.drop(columns=['1']) #include_bias=False not working in the fit_transform method. this is the workaround.

	df = dfp.copy()

	return df