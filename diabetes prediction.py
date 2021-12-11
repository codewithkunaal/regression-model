#problem statement : prediction of diabetes using linear model and plotting graph
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabeties = datasets.load_diabetes()
#  val = diabeties.keys()
# print(val)
#  dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
diabeties_x = diabeties.data[:,np.newaxis,2]
# print(diabeties_x)

#independent variable
diabeties_x_train = diabeties_x[:-50]
diabeties_x_test = diabeties_x[-50:]

#dependent variable
diabeties_y_train = diabeties.target[:-50]
diabeties_y_predicted = diabeties.target[-50:]

#creating model
model = linear_model.LinearRegression()
model.fit(diabeties_x_train,diabeties_y_train)
diabeties_y_test = model.predict(diabeties_x_test)
# print("mean squared error is :",mean_squared_error(diabeties_y_test,diabeties_y_predicted))
# print("weights :",model.coef_)
# print("intercept : ",model.intercept_)

plt.scatter(diabeties_x_test,diabeties_y_predicted)
plt.plot(diabeties_x_test,diabeties_y_test)
plt.show()
'''mean squared error is : 3035.0601152912695
weights : [941.43097333]       
intercept :  153.39713623331698'''