import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

sa = pd.read_csv(r"C:\Users\rishu\OneDrive\Desktop\rishu\Salary_Data.csv")

sa_X = sa[['YearsExperience']]
sa_Y = sa[['Salary']]

sa_X_train = sa_X[:10]

sa_X_test = sa_X[10:]

sa_Y_train = sa_Y[:10]

sa_Y_test = sa_Y[10:]


model = linear_model.LinearRegression()
model.fit(sa_X_train, sa_Y_train)

sa_Y_Predicted = model.predict(sa_X_test)


print("Mean Squared Error is: ", mean_squared_error(sa_Y_test, sa_Y_Predicted))
print("Weights ", model.coef_)
print("Intercepts ", model.intercept_)

plt.scatter(sa_X_test, sa_Y_test)
plt.plot(sa_X_test, sa_Y_Predicted)
plt.show()


