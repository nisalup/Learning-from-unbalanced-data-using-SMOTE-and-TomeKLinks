import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from plots import Plot

#load data
train_data = pd.read_csv('./input/train.csv')
train_labels = pd.read_csv('./input/train_labels.csv')
#print(train_labels.head(3))

#split train data
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, random_state=0)


unique, count = np.unique(y_train, return_counts=True)
print('counts of labels before undersampling: ', unique, count)

tl = TomekLinks(random_state=2)
X_train_res, y_train_res = tl.fit_resample(X_train, y_train.values.ravel())

unique1, count1 = np.unique(y_train_res, return_counts=True)
print('counts of labels after undersampling: ', unique1, count1)



clf = neighbors.KNeighborsClassifier(15)
clf.fit(X_train_res, y_train_res)


Z = clf.predict(X_train)
acc = clf.score(X_train, y_train)
print('Accuracy on split training data: ' + str(acc))

# Put the result into a confusion matrix
cnf_matrix_tra = confusion_matrix(y_train, Z)
print("Recall metric - split train data: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
plt.figure(1)
plt.title('Undersampling using TomeKLinks')
plt.subplot(221)
plot = Plot()
plot.plot_confusion_matrix(cnf_matrix_tra , classes=[0,1], title='Confusion matrix - train')


#now try knn on the test data
Z1 = clf.predict(X_test)
acc1 = clf.score(X_test, y_test)
print('Accuracy on split test data: ' + str(acc1))

# Put the result into a confusion matrix
cnf_matrix_test = confusion_matrix(y_test, Z1)
print("Recall metric - split test data: {}%".format(100*cnf_matrix_test[1,1]/(cnf_matrix_test[1,0]+cnf_matrix_test[1,1])))
#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))
plt.subplot(222)
plot = Plot()
plot.plot_confusion_matrix(cnf_matrix_test , classes=[0,1], title='Confusion matrix - test')


#now try for the test data provided

#first, import test data
X_test_data = pd.read_csv('./input/test.csv')

Z2 = clf.predict(X_test_data)
probas = clf.predict_proba(X_test_data)
print("test")
#Put the  labels and probabilities into a txt
pd.DataFrame(Z2).to_csv('tomeklinks_test_data_predict.csv')
pd.DataFrame(probas).to_csv('tomeklinks_test_data_predict_probabilities.csv')
plt.show()