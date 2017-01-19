import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pdb

# Reading the data from the .txt file
data = np.loadtxt(open("challenge_dataset.txt","rb") , delimiter = ",")
# Separating the read data into x and y values
x_val = data[:,0].reshape(-1,1)
y_val = data[:,1].reshape(-1,1)

# Training the LinearRegression model
model = linear_model.LinearRegression()
model.fit(x_val , y_val)

# Predicting all the X values using the regressor
prediction = model.predict(x_val)

# Calculating the error in prediction from the actual value
error = np.abs(prediction - y_val)
# Finding the average error and standard deviation
mean_error = error.mean()
std_error = error.std()


# Plotting the results
plt.scatter(x_val , y_val)
plt.plot(x_val , prediction)
plt.show()

# Plotting the prediction vs actual value graph
# Also plotting the line y=x, points closer to the line are more accurate
plt.scatter(prediction , y_val)
plt.plot(y_val , y_val)
plt.show()

print("Results:\n" + "Average Error - "+str(mean_error)+"\n" + "Standard Error Deviation - "+str(std_error))
