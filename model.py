# importing the necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor

# importing the dataset
data = pd.read_csv('boston_housing_prices.csv')
data.columns = data.columns.str.strip()  # removing the extra spaces

# using the stratified shuffle so that the column 'CHAS' will be equally distributed among the train and test
split = StratifiedShuffleSplit()
for train_index,test_index in split.split(data,data['CHAS']):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]  # saving the data into train and test

train = train_set.copy()
test = test_set.copy()

train_target = train['prices']
train.drop('prices',axis = 1,inplace = True)
test_target = test['prices']
test.drop('prices',axis = 1, inplace = True)

# scaling the data
scaler = StandardScaler()
scaler.fit(train) # using fit so that we can save the mean and variance of the scaled data later
train_transformed = scaler.transform(train)

# saving the required mean and variance so that we can use it after the deployment to scale the input data
std = np.sqrt(scaler.var_)
np.save('std.npy',std)
np.save('mean.npy',scaler.mean_)

regressor = DecisionTreeRegressor()
regressor.fit(train_transformed,train_target)

pickle.dump(regressor,open('final_model.pickle','wb'))




