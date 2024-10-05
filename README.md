# co2-predictor
It is for NASA Space app challenge 2024 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# data
lol = pd.read_csv("archive op.csv")

# Drop missing values 
lol1 = lol.dropna()  # 

# Make a copy 
lol2 = lol1.copy()

# Set the random seed
np.random.seed(87)   

# Drop the target variable from the features
x = lol2.drop("Carbon Dioxide (ppm)", axis=1)  # Features
y = lol2["Carbon Dioxide (ppm)"]  # Target

# Split the data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=70)

# 
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(x_train, y_train)

# 

regressor.score(x_test , y_test)
