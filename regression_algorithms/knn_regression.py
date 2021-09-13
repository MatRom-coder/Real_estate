import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt




class Knn:

    def __init__(self):
        self.reg = KNeighborsRegressor(n_neighbors=3)
        self.df = pd.read_csv('../Real_estate.csv')




    def split_data(self):
        
        x = self.df.drop(["No", "X1 transaction date", "X5 latitude", "X6 longitude", "Y house price of unit area"], 1)
        y = self.df["Y house price of unit area"]
        x_train, x_test , Y_train, Y_test = train_test_split(x, y , test_size = 0.2)

        return (x_train, x_test, Y_train, Y_test)


    def training_model(self):
        
        x_train = self.split_data()[0]
        Y_train = self.split_data()[1]
        
        self.reg.fit(x_train, Y_train)

    def prediction_accuracy(self):
        acc = 0  #let's initialize the accuracy value to 0
        print("accuracy: ", self.reg.score(x_test,Y_test))

        return acc

        



if __name__ == "__main__":
    ...