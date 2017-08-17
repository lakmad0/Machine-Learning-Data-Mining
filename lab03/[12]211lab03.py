import pandas as pd
import numpy as np
import unittest


class Statistics:
    def __init__(self):
        self.covarianceMatrix = pd.DataFrame()
        self.correlationMatrix = pd.DataFrame()

    def get_covariance_matrix(self,df):
        self.covarianceMatrix = pd.DataFrame(data=np.zeros(shape=(df.shape[1], df.shape[1])))
        for i in range(0, df.shape[1]):
            for j in range(0, df.shape[1]):
                if i > j:
                    self.covarianceMatrix[i][j] = self.covarianceMatrix[j][i]
                else:
                    self.covarianceMatrix[i][j] = self.covariance(df, i, j)
        return self.covarianceMatrix

    def covariance(self, df, a, b):
        nRows = df.shape[0]
        mean1 = df[a].mean()
        mean2 = df[b].mean()
        ds1 = df[a].apply(lambda x: x - mean1)
        ds2 = df[b].apply(lambda y: y - mean2)
        mul_values = (lambda x, y: (x * y))(ds1, ds2)
        return mul_values.sum()/(nRows - 1)


    def get_correlation_matrix(self, df):
        self.correlationMatrix = pd.DataFrame(data=np.zeros(shape=(df.shape[1], df.shape[1])))
        for i in range(0, df.shape[1]):
            for j in range(0, df.shape[1]):
                if i > j:
                    self.correlationMatrix[i][j] = self.correlationMatrix[j][i]
                else:
                    self.correlationMatrix[i][j] =  self.correlation(df, i, j)
        return self.correlationMatrix

    def standard_deviation(self, df, a):
        nRows = df.shape[0]
        mean = df[a].mean()
        return (df[a].apply(lambda x: (x - mean) ** 2).sum() / (nRows - 1)) ** 0.5

    def correlation(self, df, a, b):
        return self.covariance(df, a, b) / (self.standard_deviation(df, a) * self.standard_deviation(df, b))


def main():
    data = pd.read_csv('dataset/lab03Exercise.csv',names=(0,1,2,3,4))
    for i in range(0,data.shape[1]):
        data[i] = data[i].fillna(data[i].mean())

    statistics = Statistics()
    ds1 = statistics.get_covariance_matrix(data)
    ds2 = statistics.get_correlation_matrix(data)

    print "---------------covariance_matrix--------------------"
    print ds1
    print "---------------correlation_matrix-------------------"
    print ds2
    print "----------------------------------------------------"


class StatisticsTest(unittest.TestCase):
    def setUp(self):
        print "Statistics Test: setUp: begin"
        self.statistics = Statistics()
        self.df = pd.read_csv('dataset/lab03Exercise.csv', names=(0, 1, 2, 3, 4))

    def test_covariance(self):
        print "Test covariance function"
        self.assertAlmostEqual(self.statistics.covariance(self.df, 0, 1), self.df[0].cov(self.df[1]), places=3)

    def test_correlation(self):
        print "Test correlation function"
        self.assertAlmostEqual(self.statistics.correlation(self.df, 0, 1), self.df[0].corr(self.df[1]), places=3)

    def test_standard_deviation(self):
        print "Test Standard_Deviation function"
        self.assertAlmostEqual(self.statistics.standard_deviation(self.df, 0), self.df[0].std(), places=4)

    def test_get_covariance_matrix(self):
        print "Test get_covariance_matrix function"
        ds1 = self.statistics.get_covariance_matrix(self.df).as_matrix()
        ds2 = self.df.cov().as_matrix()
        for i in range(0,ds1.shape[0]):
            np.testing.assert_array_almost_equal(ds1[i], ds2[i], decimal=3)

    def test_get_correlation_matrix(self):
        print "Test get_correlation_matrix function"
        ds1 = self.statistics.get_correlation_matrix(self.df).as_matrix()
        ds2 = self.df.corr().as_matrix()
        for i in range(0, ds1.shape[0]):
            np.testing.assert_array_almost_equal(ds1[i], ds2[i], decimal=3)

    def tearDown(self):
        print "Statistics Test: tearDown: begin"

if __name__ == '__main__':
    main()
    unittest.main()



# 2. Considering the results I obtained for the exercice, I can say most of the columns are more correlated to each other except last data column.
#    Correlation between same data column is 1. Last data column(5th) shows a high correlation only with 4th data column. Correlation
#    between other columns with last column is somewhat low but not too bad. Considering all the aspects I can say there is good
#    correlation between all of these data columns.


# 3. There is a good meaning if we use correlation to derive class value instead of using mean.
#
#    eg: if( cor1,5  <  (cor2,3 + cor3,4 + cor2,4)/3) then
#           value <- 1
#        else
#           value <- 0
#        end if


