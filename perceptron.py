import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing


def perceptron(X, w, b):
  z = np.dot(X, w) + b
  # return max(0,z)
  # return 1 / (1 + np.exp(-z)) #softmax function
  return  1 if z > 0 else 0

def binary_log_loss(y_true, y_pred):
  """
  Calculates the binary log loss.

  Args:
    y_true: True labels (0 or 1).
    y_pred: Predicted probabilities (between 0 and 1).

  Returns:
    The binary log loss.
  """

  # Clip predictions to avoid numerical issues (log(0) is undefined)
  y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

  # Calculate log loss for each sample
  loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

  return loss


df = pd.read_csv("perceptron_train__data/1000002_point_data.csv",header=None)
# Create a NumPy array with the first two columns
not_normalized_train_data = df.iloc[:, :2].values

train_data = not_normalized_train_data/2000
# print(train_data.head())
# Create a NumPy array with the third column
val_data = df.iloc[:, 2].values



#Hyperparameters
w = np.array([random.random(),random.random()])  # Weights initialization random
b = np.random.randn(1) # Bias initialization random
# w = np.array([random.uniform(0,2000),random.uniform(0,2000)])  # Weights initialization random
# b = random.uniform(0,2000) # Bias initialization
# w = np.array([0.1, 5.1])  # Weights initialization manually
learning_rate = 0.1
decay_factor = 1
batch_size = 100

# For plotting
LogLossArray = np.array([0])

#this is for batch training
for p in range(0, len(train_data), batch_size):
    batch_train = train_data[p:p+batch_size]
    batch_val = val_data[p:p+batch_size]
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    Loss = 0

    for i, j in zip(batch_train, batch_val):
      output = perceptron(i, w, b)
      # print(i, 'desired', j, 'actual', output)
      dw += (j-output) * i
      db += (j-output)
      Loss += binary_log_loss(j,output)

    meanLogLoss = Loss/batch_size
    LogLossArray = np.vstack((LogLossArray, meanLogLoss))

    print('meanLogLoss Error = ', meanLogLoss)
    print('weights:', w, 'bias : ', b)
    print('dw = ',dw,'......db = ',db)
    w += learning_rate * meanLogLoss * dw/batch_size
    b += learning_rate * meanLogLoss * db/batch_size
    learning_rate *= decay_factor


print(f'm ={-w[0]/w[1]} , c = {-b/w[1]}')
print(f'w = {w}, b = {b},learning rate = {learning_rate}')
print('training Complete')





## create scatterplot of points and decision boundary
# Extract x and y coordinates
x = train_data[:, 0]
y = train_data[:, 1]
z = val_data
colors = np.where(z > 0.5, 'green', 'blue')

# Plot the points
plt.scatter(x, y,c=colors)
# Define the line's parameters (adjust as needed)
m = -w[0]/w[1]  # Slope
c = -b/w[1]  # Intercept

# Generate x values for the line
x_line = np.linspace(min(x), max(x))
y_line = m * x_line + c
plt.plot(x_line, y_line, color='red', label='y = mx + c')

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

##plotting logLoss error of batches
plt.plot(LogLossArray[1:])
plt.xlabel('training_progress')
plt.ylabel('Error Value')
plt.title('Error plot')
plt.show()

#
# ##manually verify the output
# for i in range(10) :
#   X0 = int(input("insert X0 : "))
#   X1 = int(input("insert X1 : "))
#   Xin = np.array([X0,X1])
#   output = perceptron(Xin, w, b)
#
#   print(output)