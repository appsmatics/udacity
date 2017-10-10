#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

#paring down the data
#print "reducing training data to 1/100th"
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
print "training set size=%d" % len(features_train)

from sklearn.svm import SVC
C=10000.0
kernel="rbf"
print "kernel=%s, C=%f" % (kernel, C)
clf = SVC(C=C, kernel=kernel)
t1 = time()
clf.fit(features_train, labels_train)
t2=time()
pred = clf.predict(features_test)
t3=time()
print "SVM fit time is %f" % (t2-t1) 
print "SVM predict time is %f" % (t3-t2) 

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print "Accuracy is %f" % accuracy

print "pred[10]=%d, [26]=%d, [50]=%d" % (pred[10], pred[26], pred[50])
print "predications for Chris(1)=%d" % sum(pred)

