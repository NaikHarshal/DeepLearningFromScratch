import random
# print(random.randrange(0,20))

import csv

# Specify the file name
filename = 'perceptron_train__data/1000001_point_data.csv'

# Write the data to the CSV file
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i in range(1000000):
        X = random.randrange(0,20)
        Y = random.randrange(0,20)
        C = 1 if X+Y > 10 else 0
        if random.randrange(0,20) > 5 :
            csvwriter.writerow([X,Y,C])
        else :
            pass

print(f"CSV file '{filename}' created successfully.")