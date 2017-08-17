import pandas as pd
import numpy as np
import unittest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as mat


class RegressionModels:
    def __init__(self):
        self.y_val = []
        self.predictions =[]

    def train_logistic_regression_model(self, x, y):
        y = y.astype(int).values.ravel()
        x_train, x_val, y_train, self.y_val = train_test_split(x, y, test_size=0.2)
        log_reg = LogisticRegression()
        log_reg.fit(x_train, y_train)
        self.predictions = log_reg.predict(x_val)
        return self.predictions

    def get_prediction_accuracy(self):
        return mat.accuracy_score(self.y_val, self.predictions)


def main():
    channels = pd.read_csv('dataset/lab04ExerciseChannels.csv',
                           names=('channel1', 'channel2', 'channel3', 'channel4', 'channel5'))
    angles = pd.read_csv('dataset/lab04ExerciseAngles.csv', names=('angle1', 'angle2', 'angle3'))

    data_set = pd.concat([channels, angles], axis=1)

    reg_models = RegressionModels()
    prob_values = reg_models.train_logistic_regression_model(data_set[['channel2', 'channel5']], data_set[['angle2']])
    print "--------Actual values-----------------"
    print reg_models.y_val
    print "--------Predicted values--------------"
    print prob_values
    print "--------Accuracy Score----------------"
    print reg_models.get_prediction_accuracy()


class RegressionModelsTest(unittest.TestCase):
    def setUp(self):
        print "RegressionModels Test setUp: begin"
        self.channels = pd.read_csv('dataset/lab04ExerciseChannels.csv',
                                    names=('channel1', 'channel2', 'channel3', 'channel4', 'channel5'))
        self.angles = pd.read_csv('dataset/lab04ExerciseAngles.csv', names=('angle1', 'angle2', 'angle3'))

        self.data_set = pd.concat([self.channels, self.angles], axis=1)

        reg_models = RegressionModels()
        self.prob_values = reg_models.train_logistic_regression_model(self.data_set[['channel2', 'channel5']],
                                                                    self.data_set[['angle2']])

    def test_prob_value_size(self):
        self.assertEqual(np.shape(self.prob_values)[0], round(self.data_set.shape[0] * 0.2))

    def tearDown(self):
        print "RegressionModels Test: tearDown: begin"

if __name__ == '__main__':
    main()
    unittest.main()

# 3. I selected logistic regression model to build my solution. Reason for that is linear regression model
#    is suitable for continuous value predictions but in this case the range of the output is small.
#    In a cases like this logistic regression do a better job than the linear regression model.
#    Because of that reason I selected logistic regression model

# 4. I selected channel2 and channel5 as features. First I got the correlation matrix and selected the
#    almost correlated channel sets(the sets, which have high correlation values between them). And removed the
#    one channel for each set. Channel5 has less correlations with all the other channels.
#    It is like channel5 has almost unique values when compare others. Because of that I selected channel5.
#    Channel2 have high correlation values with other channels except channel5.
#    Because of that reason channel2 can represent the other channels except channel5.
#    By considering all of these aspects I selected channel2 and channel 5 as features.
#
#    The correlations between angle2, channel2 and channel5 has the smallest difference when compared to angle1 and
#    angle3. Because of that reason I selected angle2 for predicting.