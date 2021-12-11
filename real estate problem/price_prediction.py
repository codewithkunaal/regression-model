#problem - a real time house price predictor regressive model  
#importing req models and modules 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


house = pd.read_csv("Boston.csv")
# house.keys()
# house.head

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(house,house['chas']):
    strat_train = house.loc[train_index]
    strat_test = house.loc[test_index]

housing_data =  strat_train.copy()

#making pipeline to fill missing values
x_pipeline = Pipeline([
    ("fill",SimpleImputer(strategy="median")),
    ("std_scalar",StandardScaler())
    ])

house_tr = x_pipeline.fit_transform(house)

house_features = strat_train.drop("medv",axis=1)
house_labels =strat_train["medv"].copy()

#model selection 
model = LinearRegression()
model.fit(house_features,house_labels)

#checking model on some data
some_val = house_features.iloc[5:10]
some_label = house_labels.iloc[5:10]
# PreparedData = x_pipeline.transform(some_val)
model.predict(some_val)

model.fit(house_features,house_labels)
predictions = model.predict(house_features)
mean_sq_error = mean_squared_error(house_labels,predictions)
error = np.sqrt(mean_sq_error)
print(error)

score = cross_val_score(model,house_features,house_labels,cv=10)
rmse=np.sqrt(score)
def print_score(score):
    print("Scores:", score)
    print("Mean: ", score.mean())
    print("Standard deviation: ", score.std())
print_score(rmse)

#saving the model 
from joblib import dump
dump(model, 'Boston_regression.joblib') 

#testing model on new data
dataTesting = model.predict([[0.04819,80.0,3.64,0,0.392,6.108,32.0,9.2203,1,315,16.4,392.89,6.57,]])
print(dataTesting)



