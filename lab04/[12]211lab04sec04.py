from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class RegressionModels:
    def __init__(self):
        self.iris = datasets.load_iris()
        self.x = []
        self.y = []

    def load_petal_width(self):
        self.x = self.iris.data[:, 3:]

    def load_class_values(self):
        target = self.iris.target
        for i in range(0, target.size):
            if target[i] == 2:
                self.y.append(1)
            else:
                self.y.append(0)

    def train_logistic_regression_model(self):
        x_train, x_val, y_train, y_val = train_test_split(self.x, self.y, test_size=0.2)
        log_reg = LogisticRegression()
        log_reg.fit(x_train, y_train)
        y_probability = log_reg.predict(x_val)
        return y_probability

    def train_linear_regression_model(self):
        x_train, x_val, y_train, y_val = train_test_split(self.x, self.y, test_size=0.2)
        lin_reg = LinearRegression()
        lin_reg.fit(x_train, y_train)
        y_probability = lin_reg.predict(x_val)
        return y_probability


def main():
    regressionModels = RegressionModels()
    print regressionModels.iris
    print "-----------------------------------------------------------------"
    print "-----------------------------------------------------------------"
    regressionModels.load_petal_width()
    print regressionModels.x
    print "-----------------------------------------------------------------"
    print "-----------------------------------------------------------------"
    regressionModels.load_class_values()
    print regressionModels.y
    print "-----------------------------------------------------------------"
    print "-----------------------------------------------------------------"
    linvalues = regressionModels.train_linear_regression_model()
    print linvalues
    print "-----------------------------------------------------------------"
    print "-----------------------------------------------------------------"
    logisticValues = regressionModels.train_logistic_regression_model()
    print logisticValues
    print "-----------------------------------------------------------------"
    print "-----------------------------------------------------------------"


if __name__ == '__main__':
    main()