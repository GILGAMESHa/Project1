import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time


def get_score(y_t, y_pre):
    s = []
    s.append(round(accuracy_score(y_test, y_predict), 5))  # accuracy
    s.append(round(precision_score(y_test, y_predict), 5))  # precision
    s.append(round(recall_score(y_test, y_predict), 5))  # recall
    s.append(round((2 * s[1] * s[2]) / (s[1] + s[2]), 5))  # F-measure
    return s


# read train data set
df = pd.read_csv("new_data.csv")    # Enter the path for the training set
data_set = df.values
x_train = data_set[:, 5:20]  # features
y_train = data_set[:, 4]  # labels


# read test data set
df1 = pd.read_csv("test_set.csv")   # Enter the path for the test set
test_set = df1.values
x_test = test_set[:, 5:20]
y_test = test_set[:, 4]


# Decision tree
time_start = time.time()
# start
DT = DecisionTreeClassifier(criterion='entropy')
DT.fit(x_train, y_train)            # train
y_predict = DT.predict(x_test)   # test
# end
time_end = time.time()

score = get_score(y_test, y_predict)
print("Decision tree:", '\taccuracy:', score[0], '\tprecision:', score[1],
      '\trecall:', score[2], '\tF_measure:', score[3], '\ttime cost:', time_end-time_start, 's')


# Multi-Layer Perceptron
time_start = time.time()
MLP = MLPClassifier(activation='identity')
MLP.fit(x_train, y_train)           # train
y_predict = MLP.predict(x_test)  # test
time_end = time.time()

score = get_score(y_test, y_predict)
print("MLP:", '\taccuracy:', score[0], '\tprecision:', score[1],
      '\trecall:', score[2], '\tF_measure:', score[3], '\ttime cost:', time_end-time_start, 's')


# k-Nearest Neighbor
time_start = time.time()
KNN = KNeighborsClassifier(n_neighbors=17, weights='distance', algorithm='kd_tree')
KNN.fit(x_train, y_train)            # train
y_predict = KNN.predict(x_test)  # test
time_end = time.time()

score = get_score(y_test, y_predict)
print("KNN:", '\taccuracy:', score[0], '\tprecision:', score[1],
      '\trecall:', score[2], '\tF_measure:', score[3], '\ttime cost:', time_end-time_start, 's')


# Support Vector Machine
time_start = time.time()
SVM = SVC(kernel='linear')
SVM.fit(x_train, y_train)            # train
y_predict = SVM.predict(x_test)  # test
time_end = time.time()

score = get_score(y_test, y_predict)
print("SVM:", '\taccuracy:', score[0], '\tprecision:', score[1],
      '\trecall:', score[2], '\tF_measure:', score[3], '\ttime cost:', time_end-time_start, 's')







