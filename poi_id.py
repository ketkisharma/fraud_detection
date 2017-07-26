#!/usr/bin/python

import pickle
import sys
import pandas
import numpy as np
import matplotlib.pyplot
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

# Convert dictionary to dataframe for exploratory data analysis
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

# set the index of df to be the employees series:
df.set_index(employees, inplace=True)

# Exploratory data analysis
print 'Number of data points:', len(df)
print 'Number of POI:', len(df[df['poi']==True])
print 'Number of non-POI:', len(df[df['poi']==False])
print 'Number of features available', df.shape[1]

# Test for columns if there are too many NaN values, see pdf file on insider pay to check different columns
column_name = 'director_fees'
zero_val = df[df[column_name]=='NaN']
print 'Number of NaN values in %s column: %d'%(column_name, len(zero_val))

# plot salary vs bonus to detect outliers
matplotlib.pyplot.scatter(df['salary'],df['bonus'])
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# Remove outliers and employees with no data
print 'Removing outliers and fields with NaN values'
data_dict.pop('TOTAL', 0)     # outlier
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)    # not an employee
data_dict.pop('LOCKHART EUGENE E', 0)     # has only NaN values
# remove rows with too many NaN values
df.drop('loan_advances', axis=1, inplace=True)

# replace NaN values with 0
df.replace('NaN', 0, inplace=True)

# Create new features and add to dataframe
print "Adding new features"
def computeFraction(poi_messages, all_messages):
    if poi_messages == 0 or all_messages == 0:
        fraction = 0.
    else:
        fraction = float(poi_messages)/float(all_messages)
    return fraction

fraction_from_poi = df.apply(lambda row: computeFraction(row["from_poi_to_this_person"], row["to_messages"]), axis=1)
df['fraction_from_poi'] = fraction_from_poi
fraction_to_poi = df.apply(lambda row: computeFraction(row["from_this_person_to_poi"], row["from_messages"]), axis=1)
df['fraction_to_poi'] = fraction_to_poi

# Create dictionary from the modified dataframe for further analysis
my_dataset = df.to_dict('index')

financial_features_list = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                           'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                           'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                           'director_fees']
email_features_list = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
                       'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']

# removed loan_advances and email_address from the initial features tested
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'bonus',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'restricted_stock', 'from_poi_to_this_person',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Generate labels and features from cleaned data my_dataset
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# define the steps of pipeline
scaler = MinMaxScaler()
pca = PCA()
skb = SelectKBest()
nb = GaussianNB()
dt = DecisionTreeClassifier()
svc = svm.SVC()
knc = KNeighborsClassifier()
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()

# prepare pipeline
pipe = Pipeline([('scaling', scaler,), ('pca', pca), ('skb',skb), ('Naive Bayes', nb)])
#pipe = Pipeline([('scaling', scaler),('pca', pca),('skb',skb), ('dt', dt)])
#pipe = Pipeline([('scaling', scaler),('pca', pca), ('skb',skb), ('svc', svc)])
#pipe = Pipeline([('scaling', scaler,),('pca', pca), ('skb',skb), ('knc', knc)])
#pipe = Pipeline([('scaling', scaler,),('pca', pca),('skb',skb), ('rfc', rfc)])
# pipe = Pipeline([('pca', pca),('skb',skb), ('abc', abc)])

# define param_grid for Grid Search for different algorithms

param = {'pca__n_components': range(7,13), 'pca__whiten': [True, False], 'skb__k': [2,3,4,5,6]}
'''
param = {'pca__n_components':range(6,10), 'pca__whiten': [True, False],'skb__k': [2,3,4,5],
            'svc__C': [1, 5, 10, 100, 1000], 'svc__kernel': ['rbf', 'linear','sigmoid','poly']}

param = {'pca__n_components': range(7,13), 'pca__whiten': [True, False], 'skb__k': [2,3,4,5,6],
           'dt__min_samples_split': [2, 3, 4, 5], 'dt__criterion': ['gini', 'entropy'],
           'dt__max_depth': [None, 1, 2, 3, 5, 10], 'dt__min_samples_leaf':  [1, 2, 3, 4, 5, 6, 7, 8]}

param = {'pca__n_components':range(6,10), 'pca__whiten': [True, False], 'skb__k': [2,3,4,5],
           'knc__n_neighbors': [2,3,4,5,6], 'knc__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

param = {'pca__n_components':range(6,10), 'pca__whiten': [True, False], 'skb__k': [2,3,4,5],
         'rfc__n_estimators': [10,15,20,25], 'rfc__criterion': ['gini', 'entropy']}

param = {'pca__n_components':range(6,10), 'pca__whiten': [True, False], 'skb__k': [2,3,4,5],
         'abc__base_estimator': [None], 'abc__n_estimators': [50], 'abc__learning_rate': [1.0,2.0,3.0,4.0,5.0], 
         'abc__algorithm': ['SAMME.R','SAMME'], 'abc__random_state': [None]}
'''
sss = StratifiedShuffleSplit(labels, 1000, test_size=0.2, random_state=40)
gs = GridSearchCV(pipe, param_grid=param, cv=sss, scoring='f1')
gs.fit(features, labels)

best_pipeline = gs.best_estimator_
print 'Best score: %0.3f' % gs.best_score_
print 'Best parameters set:'
best_parameters = gs.best_estimator_.get_params()
for param_name in sorted(param.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])

k_best = gs.best_estimator_.named_steps['skb']
# Get SelectKBest scores, rounded to 2 decimal places
feature_scores = ['%.2f' % elem for elem in k_best.scores_ ]
print sorted(feature_scores, reverse=True)
#steps = best_pipeline.named_steps["dt"]
#print steps.feature_importances_

# create pickle files from the model selected by GridSearchCV
print 'Creating pickle files with the model selected by GridSearchCV'
from mytester import dump_classifier_and_data
clf=gs.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)

# See the precision and recall scores using the 'test_classifier()' function;
# tester modified to mytester to specify a different random state

from mytester import test_classifier
print 'Tester Classification report'
test_classifier(clf, my_dataset, features_list)
