import math
import os

import numpy as np
import pandas as pd

def perceptron(X, w, b):
  z = np.dot(X, w) + b
  # return 1 if z > 0 else 0
  return max(0, z) , 1 if z > 0 else 0

#
# train_data = np.array([[0,0],[0,1],[1,0],[1,1]])
# val_data = np.array([0,1,1,1])
print(os.getcwd())
df = pd.read_csv("perceptron_train__data/OR_gate_data.csv",header=None)
# Create a NumPy array with the first two columns
train_data = df.iloc[:, :2].values

# Create a NumPy array with the third column
val_data = df.iloc[:, 2].values



import random
w = np.array([random.random(),-random.random()])  # Weights initialization
w = np.array([0.1, -0.6])  # Weights initialization manually


b = 0.5 # Bias
learning_rate = 0.3
decay_factor = 0.9



for i,j in zip(train_data,val_data):
  print(w,b)

  output,ans = perceptron(i, w, b)
  print(i,j,output)
  # error = math.sqrt((output - j)**2)
  error= output - j
  print(error)

  w += learning_rate * error * i
  b += learning_rate * error
  learning_rate *=decay_factor
print(-w[0]/w[1] ,-b/w[1])

print('training Complete')




## create scatterplot
import matplotlib.pyplot as plt
# Extract x and y coordinates
x = train_data[:, 0]
y = train_data[:, 1]

# Plot the points
plt.scatter(x, y)
# Define the line's parameters (adjust as needed)
m = -w[0]/w[1]  # Slope
c = -b/w[1]  # Intercept

# Generate x values for the line
x_line = np.linspace(min(x), max(x), 100)
# Calculate y values for the line
y_line = m * x_line + c
# Plot the line
plt.plot(x_line, y_line, color='red', label='y = 2x + 1')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of NumPy Array')
plt.grid(True)
plt.show()

##manually verify the ourput
for i in range(10) :
  X0 = int(input("insert X0 : "))
  X1 = int(input("insert X1 : "))
  Xin = np.array([X0,X1])
  output,ans = perceptron(Xin, w, b)

  print(ans)