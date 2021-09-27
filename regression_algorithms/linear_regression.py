import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


df = pd.read_csv("../Real_estate.csv")

x = df.drop(["No", "X1 transaction date", "X5 latitude", "X6 longitude", "Y house price of unit area"], 1)
y = df["Y house price of unit area"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

#Save the trained model to disk
filename = '../trained_models/linear_regression_model.sav'
joblib.dump(lin_reg, filename)


 