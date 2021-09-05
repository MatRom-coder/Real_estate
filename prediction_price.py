import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("./Real_estate.csv")


x = df.drop(["No", "X1 transaction date", "X1 transaction date"], 1)
y = df["Y house price of unit area"]


x_train, x_test , Y_train, Y_test = train_test_split(x, y , test_size = 0.2, shuffle = False )

model = LinearRegression()
model.fit(x_train, Y_train)
y_pred = model.predict(x_test)

print(y_pred)
print("MAE: ", mean_absolute_error(Y_test,y_pred))
 