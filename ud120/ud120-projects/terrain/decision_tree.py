#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from class_vis import output_image

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn import tree
clf2 = tree.DecisionTreeClassifier(min_samples_split=2)
clf50 = tree.DecisionTreeClassifier(min_samples_split=50)
clf2.fit(features_train, labels_train)
clf50.fit(features_train, labels_train)

try:
	print "Pretty Picture"
	prettyPicture(clf50, features_test, labels_test)
#	output_image("test.png", "png", open("test.png", "rb").read())
except NameError:
	print "Error calling prettyPicture"
	pass

pred2 = clf2.predict(features_test);
pred50 = clf50.predict(features_test);
from sklearn.metrics import accuracy_score
acc2 = accuracy_score(labels_test,pred2)
acc50 = accuracy_score(labels_test,pred50)

print "Accuracy for min_samples_split of 2 is %f" % acc2
print "Accuracy for min_samples_split of 50 is %f" % acc50





