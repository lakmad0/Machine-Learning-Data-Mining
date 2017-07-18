import numpy as np


class SnCalculater:
    csvFile = ''
    output = []

    def __init__(self, csvFile):
        self.csvFile = csvFile

    def getSn(self):
        # Read csv file
        data = np.genfromtxt(self.csvFile, delimiter=',')

        # Get dimensions of csv data
        dimension = data.shape
        nRows = dimension[0]
        nCols = dimension[1]

        # Calculate means
        means = np.mean(data, axis=0)

        # Calculate Sn
        for i in range(nCols):
            sum = 0
            for j in range(nRows):
                sum += (data[j][i] - means[i]) ** 2
            sum = (sum / (nRows - 1)) ** 0.5
            self.output.append(sum)

        return self.output


calc = SnCalculater('dataset/labExercise01.csv')
ans = calc.getSn()
print ans
