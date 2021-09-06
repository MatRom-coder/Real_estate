import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_csv("./Real_estate.csv")


"""

let'see the result considering three features -> age,distance to the nearest MRT station 
and number of convenience store.
The predicted value will be the house price of the unit area.

"""
print("FIRST TEST CONSIDERING THREE FEATURES TOGEHTER")

x = df.drop(["No", "X1 transaction date", "X5 latitude", "X6 longitude", "Y house price of unit area"], 1)
y = df["Y house price of unit area"]


x_train, x_test , Y_train, Y_test = train_test_split(x, y , test_size = 0.2, shuffle = False )

lin_reg = LinearRegression()
lin_reg.fit(x_train, Y_train)
y_pred = lin_reg.predict(x_test)
#y_pred = model.predict([[23, 23.2, 10]])

print("MAE: ", mean_absolute_error(Y_test,y_pred))
print("R^2: ", r2_score(Y_test,y_pred))


ridge_reg = Ridge(alpha = 100)
ridge_reg.fit(x_train, Y_train)
y_pred = ridge_reg.predict(x_test)

print("------------------------------------------------")
print("RIDGE REGRESSION VALUES:")
print("------------------------------------------------")
print("MAE: ", mean_absolute_error(Y_test,y_pred))
print("R^2: ", r2_score(Y_test,y_pred))
print("------------------------------------------------")


"""

let'see the result considering three features -> age,distance to the nearest MRT station 
and number of convenience store, but this time separately.
The predicted value will be the house price of the unit area.

""" 

#let' delete the unnecessary columns
new_df = df.drop(["No", "X1 transaction date", "X5 latitude", "X6 longitude", "Y house price of unit area" ], 
                 1)

i = 0
n_col = new_df.shape[1]
for _ in range(0, n_col):
    
    
    print(f"TEST WITH  {new_df.columns[i]} COLUMN: ")    
    x = new_df[new_df.columns[i]]
    x_train, x_test , Y_train, Y_test = train_test_split(x, y , test_size = 0.2, shuffle = False )
    x_train = np.array(x_train).reshape((-1, 1))
    x_test = np.array(x_test).reshape((-1, 1))
    
    
    lin_reg.fit(x_train, Y_train)
    y_pred = lin_reg.predict(x_test)
    i += 1
    
    print("MAE: ", mean_absolute_error(Y_test,y_pred))
    print("R^2: ", r2_score(Y_test,y_pred))

    
