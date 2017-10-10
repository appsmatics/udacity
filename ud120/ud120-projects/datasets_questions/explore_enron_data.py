#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print "data points=%d" % len(enron_data)
count = 0
for person in enron_data:
	if enron_data[person]["poi"]==True:
		count=count+1
	#print "%d" % len(person_data)
print "Count of pois=%d" % count

valid_salary=0
valid_email=0;
no_total_payments=0
for person in enron_data:
	if enron_data[person]['salary'] != "NaN":
		valid_salary=valid_salary+1
	if enron_data[person]['email_address'] != "NaN":
		valid_email=valid_email+1
	if enron_data[person]['total_payments'] == "NaN":
		no_total_payments=no_total_payments+1
	#print "%s total_payments=%s" % (person,enron_data[person]['total_payments'])
		
print "no_total_payments=%d" % no_total_payments

print "valid salary %d" % valid_salary
print "valid email %d" % valid_email


print len(enron_data[person])
print enron_data['SKILLING JEFFREY K']['total_payments'];
print enron_data['FASTOW ANDREW S']['total_payments'];

print enron_data['LAY KENNETH L'];



