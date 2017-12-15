#Geoffry Berryman
#Final Project
#Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import naive_bayes
from data_util import *

df = pd.read_csv('titanic.csv')
del df['ticket']
del df['cabin']
del df['boat']
del df['body']
del df['home.dest']
del df['name']

df = df.fillna(df.mean()) #Fill NA with average

#Move survived to the first column of data
cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('survived')) #Remove survived from list
df = df[['survived']+cols]

def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))


def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)

catcols = cat_features(df)
df = pd.get_dummies(df, columns=catcols)

df = df.fillna(0.0).astype(int)

data_y = df['survived']
del df['survived']
data_x = df

the_set = [[]]
num = 0	
micro = 0
macro = 0
wtd = 0
my_list = []
for item in list(data_x):
	for thing in the_set:
		the_set = the_set + [list(thing)+[item]]
		data_x = pd.DataFrame(data=df, columns=thing)
		if len(thing)>0:
			x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
			gnb_mod = naive_bayes.GaussianNB()
			gnb_mod.fit(x_train, y_train)
			preds = gnb_mod.predict(x_test)
			if str(accuracy_score(y_test, preds)) > num:
				num = str(accuracy_score(y_test, preds))
				conmat = str(confusion_matrix(y_test, preds))
				my_list = list(thing)
				micro = str(f1_score(y_test, preds, average='micro'))
				macro = str(f1_score(y_test, preds, average='macro'))
				wtd = str(f1_score(y_test, preds, average='weighted'))
				my_list = list(thing)
print('For: '+str(my_list))
print('Accuracy: '+str(num))
print('Confusion Matrix:\n'+str(conmat))
print('Micro: '+str(micro))
print('Macro: '+str(macro))
print('Weighted: '+str(wtd))
