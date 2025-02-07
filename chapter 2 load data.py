import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy
import sklearn

#HOUSING_PATH = os.path.join("datasets", "housing")

#def load_housing_data(housing_path = HOUSING_PATH):
def load_housing_data():
    return pd.read_csv("housing.csv")



housing = load_housing_data()
print(housing.head())
housing.info()  #The info method is usefult to get a quick description of the data
print(housing["ocean_proximity"].value_counts()) #Shows what categories exist and how many districts belong to each category
print(housing.describe()) #This method shows a summary of the numerical attributes

#housing.hist( bins=50, figsize=(20,15)) #shows the number of instances (vertical axis) that have a given value range
#plt.show()#Plots a histogram for each numerical attribute
