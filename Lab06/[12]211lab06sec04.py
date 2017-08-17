import numpy as np
from sklearn import datasets


iris = datasets . load_iris ()
X = iris.data
Y = iris. target

# Gaussian Naive Bayes
from sklearn . naive_bayes import GaussianNB
# clf is a classifier
clf_gnb = GaussianNB()
clf_gnb.fit(X,Y)
# Training accuracy
print ('Training Accuracy (Gaussian Naive Bayes)  : ',clf_gnb.score(X,Y))


# Multinomial Naive Bayes
from sklearn. naive_bayes import MultinomialNB
# clf_mnb is a classifier
clf_mnb = MultinomialNB()
clf_mnb.fit(X,Y)
# Training accuracy
print('Training Accuracy (Multinomial Naive Bayes) : ',clf_mnb.score(X,Y))

# Nearest Neighbor
from sklearn import neighbors
# 1-Nearest Neighbor
clf_nn = neighbors . KNeighborsClassifier ( n_neighbors =1)
clf_nn.fit(X,Y) # Model
# Training accuracy
print('Training Accuracy(Nearest Neighbor) : ',clf_nn.score(X,Y))

# Support Vector Machine
from sklearn import svm
# clf is a classifier
clf_svm = svm.SVC(kernel='linear', C=1, gamma =1). fit(X, Y)
clf_svm.fit(X,Y) # Model
# Training accuracy
print('Training Accuracy(Support Vector Machine) : ',clf_svm.score(X,Y))


# split data as 2/3 for training and 1/3 for testing
from sklearn. model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split (X, Y, test_size =0.333 , random_state =0)
clf_nn.fit( X_train , Y_train)
print('Test Accuracy: ',clf_nn.score(X_test , Y_test))

# Use cross_val_score helper function for cross validation
from sklearn. model_selection import cross_val_score
# 10−fold cross validation
scores = cross_val_score (clf_nn , X, Y, cv =10)
print(scores) # Results for all the folds
# print("10CV Accuracy: %0.2f (+/− %0.2f)" % (scores.mean (), (scores.std ()∗2)))
