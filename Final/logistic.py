#Logistic

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

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
prec = 0
knum = 0
my_list = []
for item in list(data_x):
	for thing in the_set:
		the_set = the_set + [list(thing)+[item]]
		data_x = pd.DataFrame(data=df, columns=thing)
		if len(thing)>0:
			x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
			log_mod = linear_model.LogisticRegression()
			log_mod.fit(x_train, y_train)
			preds = log_mod.predict(x_test)
			pred_probs = log_mod.predict_proba(x_test)
			prob_pos = pred_probs.transpose()[1]  
			prob_neg = pred_probs.transpose()[0]  
			pred_df = pd.DataFrame({'Actual':y_test, 'Predicted Class':preds, 'P(1)':prob_pos, 'P(0)':prob_neg})
			if str(accuracy_score(y_test, preds)) > num:
				if str(precision_score(y_test, preds)) > prec:
					prec = str(precision_score(y_test, preds))
					num = str(accuracy_score(y_test, preds))
					conmat = str(confusion_matrix(y_test, preds))
					recall = str(recall_score(y_test, preds))
					F1 = str(f1_score(y_test, preds))
					ROC = str(roc_auc_score(y_test, preds))
					my_list = list(thing)
print('For: '+str(my_list))
print('Accuracy: '+str(num))
print('Precision: '+str(prec))
print("Confusion Matrix:\n" + str(conmat))
print('Recall: '+str(recall))
print('F1: '+str(F1))
print('Roc: '+str(ROC))
