import random
# print(random.randrange(0,20))

import csv

# Specify the file name
filename = 'perceptron/perceptron_train__data/1000002_point_data.csv'

# Write the data to the CSV file
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i in range(1000000):
        X = random.randrange(0,2000)
        Y = random.randrange(0,2000)
        if X + 2*Y > 2000 :
            C = 1
        else:
            C = 0
        if random.randrange(0,20) > 15 :
            csvwriter.writerow([X,Y,C])
        else :
            pass

print(f"CSV file '{filename}' created successfully.")