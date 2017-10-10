#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
### it's all yours from here forward!
from sklearn.model_selection import train_test_split
data_split = train_test_split(features, labels, test_size=0.3, random_state=42)
features_train, features_test, labels_train, labels_test = data_split

#print (features_train)
#print (features_test)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc = accuracy_score(labels_test, pred)
print "Accuracy is %f" % acc

precision = precision_score(labels_test, pred)
print "Precision is %f" % precision

recall = recall_score(labels_test, pred)
print "Recall is %f" % recall

#accuracy of zeros predicted for all
zeros = [0]*29
acc = accuracy_score(labels_test, zeros)
print "Accuracy for all zeros predicted is %f" % acc



