import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing


#read in data
df = pd.read_csv('./Assignment-2/Assignment-2/AmesHousingSetA.csv')
TEST = pd.read_csv('./Assignment-2/Assignment-2/AmesHousingSetB.csv')

#Deleting columns to build better model
del df['Fence']
del df['Mo.Sold']
del df['Yr.Sold']
del df['Misc.Val']
del df['Pool.Area']
del df['X3Ssn.Porch']
del df['MS.SubClass']
del df['BsmtFin.SF.2']
del df['Screen.Porch']
del df['Enclosed.Porch']
del df['Bedroom.AbvGr']
del df['Kitchen.AbvGr']
del df['Sale.Condition']
del df['Roof.Matl']
del df['Exterior.1st']
del df['Exterior.2nd']
del df['Condition.2']
del df['Sale.Type']
del df['Pool.QC']
del TEST['Pool.QC']
del TEST['Sale.Type']
del TEST['Condition.2']
del TEST['Exterior.2nd']
del TEST['Exterior.1st']
del TEST['Roof.Matl']
del TEST['Fence']
del TEST['Mo.Sold']
del TEST['Yr.Sold']
del TEST['Misc.Val']
del TEST['Pool.Area']
del TEST['X3Ssn.Porch']
del TEST['MS.SubClass']
del TEST['BsmtFin.SF.2']
del TEST['Screen.Porch']
del TEST['Enclosed.Porch']
del TEST['Bedroom.AbvGr']
del TEST['Kitchen.AbvGr']
del TEST['Sale.Condition']

#Move SalePrice to the first column of data
cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('SalePrice')) #Remove SalePrice from list
df = df[['SalePrice']+cols]

cols2 = list(TEST.columns.values)
cols2.pop(cols2.index('SalePrice')) 
TEST = TEST[['SalePrice']+cols2]


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

#deleting columns to make training set fit test set
del df['MS.Zoning_A (agr)']
del df['Utilities_NoSeWa']
del df['Utilities_NoSewr']
del df['Neighborhood_GrnHill']
del df['Neighborhood_Landmrk']
del df['Condition.1_RRNe']
del df['Garage.Cond_Ex']
del df['Misc.Feature_Elev']
del df['Bsmt.Qual_Po']
del df['Bsmt.Cond_Ex']
del df['Heating_Floor']
del df['Heating_OthW']
del df['Electrical_Mix']
del df['Functional_Maj2']
del df['Functional_Sal']
del df['Functional_Sev']
del df['Bsmt.Cond_Po']


features = list(df)
features.remove('SalePrice')

#Training x set
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_x = imp.fit_transform(df[features])

#Official test x set
TEST_x = imp.fit_transform(TEST2[features])

#Pearson Correlation
#other = df.corr()
#print (other.head())

#training y set
data_y = df['SalePrice'] 

#official test y set
TEST_y = TEST['SalePrice']


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
linear_mod = linear_model.LinearRegression()

#Build model
linear_mod.fit(x_train,y_train)

#Predict on housing set a
#preds = linear_mod.predict(x_test)

#print('r2: ' + r2_score(TEST_y, preds))

#Predict on set B
preds = linear_mod.predict(TEST_x)


#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(TEST_y, preds), \
							   median_absolute_error(TEST_y, preds), \
							   r2_score(TEST_y, preds), \
							   explained_variance_score(TEST_y, preds)])) 


