import math
import os
import numpy as np
import pandas as pd

def perceptron(X, w, b):
  z = np.dot(X, w) + b
  return max(z,0)
  # return  1 if z > 0 else 0

#
# train_data = np.array([[0,0],[0,1],[1,0],[1,1]])
# val_data = np.array([0,1,1,1])
print(os.getcwd())
df = pd.read_csv("perceptron_train__data/100000_point_data.csv",header=None)
# Create a NumPy array with the first two columns
train_data = df.iloc[:, :2].values

# Create a NumPy array with the third column
val_data = df.iloc[:, 2].values



import random
#Hyperparameters
w = np.array([random.random(),random.random()])  # Weights initialization random
b = random.random() # Bias initialization random
# w = np.array([random.uniform(0,20),random.uniform(0,20)])  # Weights initialization random
# b = random.randrange(0,20) # Bias initialization
# w = np.array([10.0, 0.1])  # Weights initialization manually
# b = 20 # Bias initialization manually
learning_rate = 0.9
decay_factor = 0.9
batch_size = 100

# For plotting
Biases = w
RMSE_array = np.array([0])

for p in range(0, len(train_data), batch_size):
    batch_train = train_data[p:p+batch_size]
    batch_val = val_data[p:p+batch_size]
    error = 0
    RMSE = 0
    for i, j in zip(batch_train, batch_val):
      output = perceptron(i, w, b)
      # print(i, 'desired', j, 'actual', output)
      error += (j - output)
      RMSE += (j-output)**2
    RMSE = math.sqrt(RMSE)/100
    RMSE_array = np.vstack((RMSE_array, RMSE))

    print('RMSE Error = ', RMSE)

    print('weights:', w, 'bias : ', b)
    print(np.mean(batch_train,axis=0))
    Biases = np.vstack((Biases, w))
    if error >= 0 :
      w += learning_rate * RMSE * np.mean(batch_train,axis=0)
      b += learning_rate * RMSE
    else :
      w -= learning_rate * RMSE * np.mean(batch_train,axis=0)
      b -= learning_rate * RMSE
    learning_rate *= decay_factor
#
# for i,j in zip(train_data,val_data):
#   print('weights:',w,'bias : ', b)
#   Biases = np.vstack((Biases,w))
#   output = perceptron(i, w, b)
#   print(i,'desired',j,'actual',output)
#   # error = math.sqrt((output - j)**2)
#   error= j-output
#   print('Error = ',error)
#
#   w += learning_rate * error * i
#
#   b += learning_rate * error
#
#   learning_rate *=decay_factor
print(f'm ={-w[0]/w[1]} , c = {-b/w[1]}')
print(f'w = {w}, b = {b},learning rate = {learning_rate}')
print('training Complete')



import matplotlib.pyplot as plt

## create scatterplot of points and decision boundary
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

# #create satterplot of weights
# x = Biases[:, 0]
# y = Biases[:, 1]
# plt.scatter(x,y)
#
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Scatter Plot of NumPy Array')
# plt.grid(True)
# plt.show()

# create scatterplot of errors (RMSE_list)

# Plot the array
plt.plot(RMSE_array)
plt.xlabel('training_progress')
plt.ylabel('Error Value')
plt.title('Error plot')
plt.show()


##manually verify the ourput
for i in range(10) :
  X0 = int(input("insert X0 : "))
  X1 = int(input("insert X1 : "))
  Xin = np.array([X0,X1])
  output = perceptron(Xin, w, b)

  print(output)