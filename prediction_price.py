import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


df = pd.read_csv("./Real_estate.csv")


x = df.drop(["No", "X1 transaction date", "X5 latitude", "X6 longitude", "Y house price of unit area"], 1)
y = df["Y house price of unit area"]


x_train, x_test , Y_train, Y_test = train_test_split(x, y , test_size = 0.2, shuffle = False )

model = LinearRegression()
model.fit(x_train, Y_train)
y_pred = model.predict(x_test)




print(y_pred)
print("MAE: ", mean_absolute_error(Y_test,y_pred))
print("R^2: ", r2_score(Y_test,y_pred))

 