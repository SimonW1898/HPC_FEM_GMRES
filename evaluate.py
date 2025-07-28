import csv
import matplotlib.pyplot as plt
import numpy as np


def readCSVFile():
    variables = {}

    with open('data_output/data.csv', 'r') as file:
        reader = csv.reader(file)

        for row in reader:
            variable_name = row[0]
            values = [float(value) for value in row[1:]]

            variables[variable_name] = values

    return variables


def read_rig_file(file_path):
    matrix = None
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file):
            line = line.strip()
            if line_num == 0:
                dimensions = list(map(int, line.split()))[0:2]
                matrix = np.zeros(dimensions)
            else:
                i, j, value = map(float, line.split())
                i = int(i) - 1
                j = int(j) - 1
                matrix[i, j] = value
    return matrix

# Example usage
# file_path = 'mat63.rig'  # Replace with the actual file path
# matrix = read_rig_file(file_path)
# print(matrix)


data = readCSVFile()
resvec = data['resvec']
plt.semilogy(resvec)
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Residual Vector')
plt.savefig('data_output/residual.jpg')
