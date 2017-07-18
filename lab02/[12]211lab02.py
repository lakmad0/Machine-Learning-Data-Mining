import math
import matplotlib.pyplot as plt
import numpy as np
import unittest


class GradientDescentAlgorithm:
    def __init__(self, e1, e2, e3):
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def get_function_values(self, x):
        fx = [(self.e2 * (math.pow(x[i], self.e1 % 5)) - self.e3 * (math.pow(x[i], self.e2 % 5)) - self.e1 * (
            math.pow(x[i], self.e3 % 5)) + self.e3) for i in range(0, len(x))]
        return fx

    def get_derivative_values(self, x):
        derivative = [(self.e2 * (self.e1 % 5) * (math.pow(x[i], ((self.e1 % 5) - 1)))
                       - self.e3 * (self.e2 % 5) * (math.pow(x[i], ((self.e2 % 5) - 1)))
                       - self.e1 * (self.e3 % 5) * (math.pow(x[i], ((self.e3 % 5) - 1)))) for i in range(0, len(x))]
        return derivative

    def gradient_decent_algo(self, initial_value, learning_rate, precision):
        step_size = 99999999
        current_value = initial_value

        while step_size > precision:
            previous_value = current_value
            current_value = current_value - learning_rate * (self.get_derivative_values([current_value])[0])
            step_size = abs(current_value - previous_value)

        return current_value


def main():
    algorithm = GradientDescentAlgorithm(2, 1, 1)
    x_dim = np.linspace(-10, 10, num=1000)
    fx = algorithm.get_function_values(x_dim)
    plt.plot(x_dim, fx)
    plt.show()

    local_minima = algorithm.gradient_decent_algo(10, 0.4, 0.000001)
    print local_minima


class GradientDescentAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.algo = GradientDescentAlgorithm(2, 1, 1)
        print "Gradient Descent Algorithm Test: setUp: begin"

    def test_function_values(self):
        self.assertEquals(self.algo.get_function_values([-1, 0, 1]), [5, 1, -1])

    def test_derivative_values(self):
        self.assertEquals(self.algo.get_derivative_values([-1, 0, 1]), [-5, -3, -1])

    def test_local_minima_value(self):
        eps = 0.01
        local_minima = self.algo.gradient_decent_algo(10, 0.001, 0.000001)
        self.assertGreater(self.algo.get_derivative_values([local_minima + eps])[0], 0)
        self.assertLess(self.algo.get_derivative_values([local_minima - eps])[0], 0)
        self.assertAlmostEqual(self.algo.get_derivative_values([local_minima])[0], 0, places=2)

    def tearDown(self):
        print "Gradient Descent Algorithm Test: tearDown: begin"

if __name__ == '__main__':
    main()
    unittest.main()
