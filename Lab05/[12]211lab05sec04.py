import pandas as pd
import numpy as np
from sklearn import tree
import sklearn.metrics as mat
from sklearn.model_selection import train_test_split
import pydotplus
from IPython.display import Image as im
from PIL import Image
import unittest


class DecisionTrees:
    def __init__(self, data_set):
        self.data_set = data_set.dropna()
        self.classifier = tree.DecisionTreeClassifier()
        self.predictors = []
        self.target = []
        self.tar_test = []
        self.predictions =[]

    def convert2numeric(self):
        self.data_set['ALCCONSUMPTION'] =  pd.to_numeric(self.data_set['ALCCONSUMPTION'], errors='coerce')
        self.data_set['BREASTCANCERPER100TH'] = pd.to_numeric(self.data_set['BREASTCANCERPER100TH'], errors='coerce')
        self.data_set['FEMALEEMPLOYRATE'] = pd.to_numeric(self.data_set['FEMALEEMPLOYRATE'], errors='coerce')

    def bin2data(self):
        self.data_set['BIN2ALCOHOL'] = self.data_set['ALCCONSUMPTION'].apply(
            lambda x: 1 if x >=5 else 0)
        self.data_set['BIN2CANCER'] = self.data_set['BREASTCANCERPER100TH'].apply(
            lambda x: 1 if x >= 20 else 0)
        self.data_set['BIN2FEMALEEMPLOYEE'] = self.data_set['FEMALEEMPLOYRATE'].apply(
            lambda x: 1 if x >= 50 else 0)

    def predict_cancer(self):
        test_ratio = 1.0/3.0
        self.predictors = self.data_set[['ALCCONSUMPTION', 'FEMALEEMPLOYRATE']]
        self.target = self.data_set[['BIN2CANCER']]
        pred_train, pred_test, tar_train, self.tar_test = train_test_split(self.predictors, self.target, test_size=test_ratio)
        self.classifier = self.classifier.fit(pred_train, tar_train)
        self.predictions = self.classifier.predict(pred_test)
        return self.predictions

    def get_prediction_accuracy(self):
        return mat.accuracy_score(self.tar_test, self.predictions)

    def print_output(self):
        dot_data = tree.export_graphviz(self.classifier, out_file=None, filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        im(graph.create_png())
        graph.write_png("cancer.png")
        graph.write_pdf("cancer.pdf")
        image = Image.open("cancer.png")
        image.show()


def main():
    data_with_na = pd.read_csv('dataset/breaset-cancer.csv', header=0)

    # handle the missing values
    data_set = data_with_na.dropna()

    # Initializing DecisionTree Object
    decision_tree  = DecisionTrees(data_set)
    decision_tree.convert2numeric()
    decision_tree.bin2data()
    prediction = decision_tree.predict_cancer()
    print "-------------- Actual Values -----------------"
    print decision_tree.tar_test['BIN2CANCER'].tolist()
    print "-------------- Predicted Values --------------"
    print prediction.tolist()
    print "------------- Prediction Accuracy ------------"
    print decision_tree.get_prediction_accuracy()

    # get png/pdf file
    decision_tree.print_output()


class DecisionTreesTest(unittest.TestCase):
    def setUp(self):
        print "DecisionTrees Test setUp: begin"
        data_with_na = pd.read_csv('dataset/breaset-cancer.csv', header=0)
        # handle the missing values
        data_set = data_with_na.dropna()
        # Initializing DecisionTree Object
        self.decision_tree = DecisionTrees(data_set)

    def test_convert2numeric_function(self):
        self.decision_tree.convert2numeric()
        self.assertEqual(type(self.decision_tree.data_set['ALCCONSUMPTION'][5]),np.float64)
        self.assertEqual(type(self.decision_tree.data_set['BREASTCANCERPER100TH'][5]), np.float64)
        self.assertEqual(type(self.decision_tree.data_set['FEMALEEMPLOYRATE'][5]), np.float64)

    def test_bin2data_function(self):
        self.assertFalse(
            set(['BIN2ALCOHOL', 'BIN2CANCER', 'BIN2FEMALEEMPLOYEE']).issubset(self.decision_tree.data_set.columns))
        self.decision_tree.bin2data()
        self.assertTrue(
            set(['BIN2ALCOHOL', 'BIN2CANCER', 'BIN2FEMALEEMPLOYEE']).issubset(self.decision_tree.data_set.columns))

    def tearDown(self):
        print "DecisionTrees Test: tearDown: begin"


if __name__ == '__main__':
    main()
    unittest.main()

# Assumption: Only alcohol consumption & female employee rate are effect for the breast cancer
