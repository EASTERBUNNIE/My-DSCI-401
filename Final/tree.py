#Decision Tree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from data_util import *

data = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')

del data['ticket']
del data['cabin']
del data['boat']
del data['body']
del data['home.dest']
del data['name']
del test['ticket']
del test['cabin']
del test['boat']
del test['body']
del test['home.dest']
del test['name']

test = test.fillna(test.mean())
data = data.fillna(data.mean())

#Move survived to the first column of data
cols = list(data.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('survived')) #Remove survived from list
df = data[['survived']+cols]

cols2 = list(test.columns.values) #Make a list of all of the columns in the df
cols2.pop(cols2.index('survived')) #Remove survived from list
test = test[['survived']+cols2]

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
TEST2 = pd.get_dummies(test, columns=catcols)

df = df.fillna(0.0).astype(int)
test = df.fillna(0.0).astype(int)

data_y = df['survived']
del df['survived']
data_x = df
test_y = TEST2['survived']
del TEST2['survived']
test_x = TEST2

the_set = [[]]
num = 0	
micro = 0
macro = 0
wtd = 0
my_list = []
mod = 0
for item in list(data_x):
	for thing in the_set:
		the_set = the_set + [list(thing)+[item]]
		data_x = pd.DataFrame(data=df, columns=thing)
		if len(thing)>0:
			x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
			dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
			dtree_gini_mod.fit(x_train, y_train)
			preds = dtree_gini_mod.predict(x_test)
			if str(accuracy_score(y_test, preds)) > num:
				num = str(accuracy_score(y_test, preds))
				conmat = str(confusion_matrix(y_test, preds))
				my_list = list(thing)
				micro = str(f1_score(y_test, preds, average='micro'))
				macro = str(f1_score(y_test, preds, average='macro'))
				wtd = str(f1_score(y_test, preds, average='weighted'))
				my_list = list(thing)
				model_type = 'gini'
				mod = 0
			dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
			dtree_entropy_mod.fit(x_train, y_train)
			preds = dtree_entropy_mod.predict(x_test)
			if str(accuracy_score(y_test, preds)) > num:
				num = str(accuracy_score(y_test, preds))
				conmat = str(confusion_matrix(y_test, preds))
				my_list = list(thing)
				micro = str(f1_score(y_test, preds, average='micro'))
				macro = str(f1_score(y_test, preds, average='macro'))
				wtd = str(f1_score(y_test, preds, average='weighted'))
				my_list = list(thing)
				model_type = 'entropy'
				mod = 1
print('For: '+str(my_list))
print('Model type: '+str(model_type))
print('Accuracy: '+str(num))
print('Confusion Matrix:\n'+str(conmat))
print('Micro: '+str(micro))
print('Macro: '+str(macro))
print('Weighted: '+str(wtd))

data_x = pd.DataFrame(df, columns=my_list)
test_x = pd.DataFrame(test_x, columns=my_list)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
if mod == 0:
	dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
	dtree_gini_mod.fit(x_train, y_train)
	preds = dtree_gini_mod.predict(test_x)
	num = str(accuracy_score(test_y, preds))
	conmat = str(confusion_matrix(test_y, preds))
	micro = str(f1_score(test_y, preds, average='micro'))
	macro = str(f1_score(test_y, preds, average='macro'))
	wtd = str(f1_score(test_y, preds, average='weighted'))
if mod == 1:
	dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
	dtree_entropy_mod.fit(x_train, y_train)
	preds = dtree_gini_mod.predict(test_x)
	num = str(accuracy_score(testy, preds))
	conmat = str(confusion_matrix(test_y, preds))
	micro = str(f1_score(test_y, preds, average='micro'))
	macro = str(f1_score(test_y, preds, average='macro'))
	wtd = str(f1_score(test_y, preds, average='weighted'))
print('For: '+str(my_list))
print('Model type: '+str(model_type))
print('Accuracy: '+str(num))
print('Confusion Matrix:\n'+str(conmat))
print('Micro: '+str(micro))
print('Macro: '+str(macro))
print('Weighted: '+str(wtd))