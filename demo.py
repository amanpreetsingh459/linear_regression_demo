import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

prediction_result = body_reg.predict(14.164)
print("Prediction result: ", prediction_result)

actual_result = dataframe.iloc[0][1]
print("Actual result: ", actual_result)

prediction_error = actual_result - prediction_result
print("Prediction error: ", prediction_error)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()