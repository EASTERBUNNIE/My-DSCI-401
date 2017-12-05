#Geoffry Berryman

import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

#read in data
df = pd.read_csv('./Assignment-3/Assignment-3/churn_data.csv')
TEST = pd.read_csv('./Assignment-3/Assignment-3/churn_validation.csv')

#Move Churn to the first column of data
cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Churn')) #Remove Churn from list
df = df[['Churn']+cols]
cols2 = list(TEST.columns.values)#Same except for test data
cols2.pop(cols2.index('Churn')) 
TEST = TEST[['Churn']+cols2]

#del df['Income']
#del TEST['Income']

def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))


def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)

catcols = cat_features(df)
TEST2 = pd.get_dummies(TEST, columns=catcols)
df = pd.get_dummies(df, columns=catcols)

#Drop churn_no column to have 1 column for data_y
del df['Churn_No']
del TEST2['CustID']
del df['CustID']
del TEST2['Churn_No']

data_y = df['Churn_Yes']
del df['Churn_Yes']
data_x = df
test_y = TEST2['Churn_Yes']
del TEST2['Churn_Yes']
test_x = TEST2

the_set = [[]]
num = 0	
prec = 0
knum = 0
my_list = []
for item in list(data_x):
	for thing in the_set:
		the_set = the_set + [list(thing)+[item]]
		data_x = pd.DataFrame(data=df, columns=thing)
		x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
		ks = [2, 3, 4, 5, 6, 7, 8,9,10,12,14,16]
		if len(thing)>0:
			for k in ks:
				mod = neighbors.KNeighborsClassifier(n_neighbors=k)
				mod.fit(x_train, y_train)
				preds = mod.predict(x_test)
				if str(accuracy_score(y_test, preds)) > num:
					if str(precision_score(y_test, preds)) > prec:
						prec = str(precision_score(y_test, preds))
						num = str(accuracy_score(y_test, preds))
						knum = k
						conmat = str(confusion_matrix(y_test, preds))
						my_list = list(thing)
print('Number of Neighbors: '+str(knum))
print('For: '+str(my_list))
print('Accuracy: '+str(num))
print('Precision: '+str(prec))
print("Confusion Matrix:\n" + str(conmat))

#Test on Validation data now that we have best KNN
data_x = pd.DataFrame(df, columns=my_list)
valid = pd.DataFrame(TEST2, columns=my_list)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
mod=neighbors.KNeighborsClassifier(n_neighbors=knum)
mod.fit(x_train, y_train)
preds = mod.predict(valid)
print("-----------------Testing Churn Validation-------------------")
print('Accuracy: ' + str(accuracy_score(test_y, preds)))
print('Precison: ' + str(precision_score(test_y, preds)))
print('Recall: ' + str(recall_score(test_y, preds)))
print('F1: ' + str(f1_score(test_y, preds)))
print("Confusion Matrix:\n" + str(confusion_matrix(test_y, preds)))




