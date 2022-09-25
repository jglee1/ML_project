# Classification with Python

# Objectives
# After completing this lab you will be able to:
#
# Confidently create classification models
# In this notebook we try to practice all the classification algorithms that we learned in this course.
#
# We load a dataset using Pandas library, apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
#
# Let's first load required libraries:

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing


# About dataset

# This dataset is about the performance of basketball teams.
# The cbb.csv data set includes performance data about five seasons of 354 basketball teams.
# It includes the following fields:
#
# Field	Description
# TEAM	The Division I college basketball school
# CONF	The Athletic Conference in which the school participates in (A10 = Atlantic 10, ACC = Atlantic Coast Conference, AE = America East, Amer = American, ASun = ASUN, B10 = Big Ten, B12 = Big 12, BE = Big East, BSky = Big Sky, BSth = Big South, BW = Big West, CAA = Colonial Athletic Association, CUSA = Conference USA, Horz = Horizon League, Ivy = Ivy League, MAAC = Metro Atlantic Athletic Conference, MAC = Mid-American Conference, MEAC = Mid-Eastern Athletic Conference, MVC = Missouri Valley Conference, MWC = Mountain West, NEC = Northeast Conference, OVC = Ohio Valley Conference, P12 = Pac-12, Pat = Patriot League, SB = Sun Belt, SC = Southern Conference, SEC = South Eastern Conference, Slnd = Southland Conference, Sum = Summit League, SWAC = Southwestern Athletic Conference, WAC = Western Athletic Conference, WCC = West Coast Conference)
# G	Number of games played
# W	Number of games won
# ADJOE	Adjusted Offensive Efficiency (An estimate of the offensive efficiency (points scored per 100 possessions) a team would have against the average Division I defense)
# ADJDE	Adjusted Defensive Efficiency (An estimate of the defensive efficiency (points allowed per 100 possessions) a team would have against the average Division I offense)
# BARTHAG	Power Rating (Chance of beating an average Division I team)
# EFG_O	Effective Field Goal Percentage Shot
# EFG_D	Effective Field Goal Percentage Allowed
# TOR	Turnover Percentage Allowed (Turnover Rate)
# TORD	Turnover Percentage Committed (Steal Rate)
# ORB	Offensive Rebound Percentage
# DRB	Defensive Rebound Percentage
# FTR	Free Throw Rate (How often the given team shoots Free Throws)
# FTRD	Free Throw Rate Allowed
# 2P_O	Two-Point Shooting Percentage
# 2P_D	Two-Point Shooting Percentage Allowed
# 3P_O	Three-Point Shooting Percentage
# 3P_D	Three-Point Shooting Percentage Allowed
# ADJ_T	Adjusted Tempo (An estimate of the tempo (possessions per 40 minutes) a team would have against the team that wants to play at an average Division I tempo)
# WAB	Wins Above Bubble (The bubble refers to the cut off between making the NCAA March Madness Tournament and not making it)
# POSTSEASON	Round where the given team was eliminated or where their season ended (R68 = First Four, R64 = Round of 64, R32 = Round of 32, S16 = Sweet Sixteen, E8 = Elite Eight, F4 = Final Four, 2ND = Runner-up, Champion = Winner of the NCAA March Madness Tournament for that given year)
# SEED	Seed in the NCAA March Madness Tournament
# YEAR	Season


# Load Data From CSV File
# Let's load the dataset [NB Need to provide link to csv file]

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%206/cbb.csv')
df.head()

print(df.shape)

# Add Column¶
# Next we'll add a column that will contain "true" if the wins above bubble are over 7 and "false" if not.
# We'll call this column Win Index or "windex" for short.

df['windex'] = np.where(df.WAB > 7, 'True', 'False')

# Data visualization and pre-processing

# Next we'll filter the data set to the teams that made the Sweet Sixteen, the Elite Eight, and the Final Four in the post season.
# We'll also create a new dataframe that will hold the values with the new column.

postseason_set = set(df['POSTSEASON'].values.tolist())
print("postseason_set")
print(postseason_set)

df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
print(df1.head())

print(df1['POSTSEASON'].value_counts())

# Lets plot some columns to underestand the data better:

import seaborn as sns

bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# Pre-processing: Feature selection/extraction

# Lets look at how Adjusted Defense Efficiency plots

bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# We see that this data point doesn't impact the ability of a team to get into the Final Four.


# Convert Categorical features to numerical values

# Lets look at the postseason:

print(df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True))

# 13% of teams with 6 or less wins above bubble make it into the final four while 17% of teams with 7 or more do.
#
# Lets convert wins above bubble (winindex) under 7 to 0 and over 7 to 1:

df1['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
print(df1.head())

# Feature selection

# Let's define feature sets, X:

X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
print(X[0:5])

# What are our lables?
# Round where the given team was eliminated or where their season ended (R68 = First Four, R64 = Round of 64, R32 = Round of 32, S16 = Sweet Sixteen, E8 = Elite Eight, F4 = Final Four, 2ND = Runner-up, Champion = Winner of the NCAA March Madness Tournament for that given year)|

y = df1['POSTSEASON'].values
y[0:5]


# Normalize Data

# Data Standardization gives data zero mean and unit variance (technically should be done after train test split )

X= preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

# Training and Validation

# Split the data into Training and Validation data.

# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_test.shape,  y_test.shape)

# Classification

# Now, it is your turn, use the training set to build an accurate model. Then use the validation set to report the accuracy of the model You should use the following algorithm:
#
# K Nearest Neighbor(KNN)
# Decision Tree
# Support Vector Machine
# Logistic Regression


# K Nearest Neighbor(KNN)

# Question 1 Build a KNN model using a value of k equals five, find the accuracy on the validation data (X_val and y_val)
#
# You can use  accuracy_score

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Training

k = 5
# Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
print(neigh)

# Predicting

yhat = neigh.predict(X_test)
print(yhat[0:5])

# Accuracy evaluation

# In multilabel classification, accuracy classification score is a function that computes subset accuracy.
# This function is equal to the jaccard_score function.
# Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.


print("Train set Accuracy: ", accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_test, yhat))


# Question 2 Determine and print the accuracy for the first 15 values of k on the validation data:

Ks = 15
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

print(mean_acc)

# Plot the model accuracy for a different number of neighbors.

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

# Decision Tree

# The following lines of code fit a DecisionTreeClassifier:

from sklearn.tree import DecisionTreeClassifier

# Modeling

# We will first create an instance of the DecisionTreeClassifier called dTree.
# Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.

dTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
print(dTree)

# Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset
dTree.fit(X_train, y_train)

# Prediction

# Let's make some predictions on the testing dataset and store it into a variable called predTree.

predTree = dTree.predict(X_test)

print(predTree[0:5])
print(y_test[0:5])

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

import sklearn.tree as tree

tree.plot_tree(dTree)
plt.show()


# Question 3 Determine the minumum value for the parameter max_depth that improves results

Ks = 15
acc_array = np.zeros((Ks - 1))

for k in range(1, Ks):
    dTree = DecisionTreeClassifier(criterion='entropy', max_depth=k)

    # Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset
    dTree.fit(X_train, y_train)

    # Prediction
    predTree = dTree.predict(X_test)

    # Evaluation
    acc_array[k - 1] = metrics.accuracy_score(y_test, predTree)

print(acc_array)

print("The best accuracy was with", acc_array.max(), "with k=", acc_array.argmax() + 1)


# Support Vector Machine

# Question 4 Train the support vector machine model and determine the accuracy on the validation data for each kernel.
# Find the kernel (linear, poly, rbf, sigmoid) that provides the best score on the validation data and train a SVM using it.

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)

    print(clf)

    # After being fitted, the model can then be used to predict new values:
    yhat = clf.predict(X_test)
    print(yhat[0:5])
    print("kernel: ", kernel)
    print(classification_report(y_test, yhat))


# Logistic Regression

# Question 5 Train a logistic regression model and determine the accuracy of the validation data (set C=0.01)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss

sag_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

for solver in sag_list:
    print("LogisticRegression solver: ", solver)
    LR = LogisticRegression(C=0.01, solver=solver).fit(X_train, y_train)
    yhat = LR.predict(X_test)

    # print(jaccard_score(y_test, yhat, pos_label=7))
    print("jaccard_score -")
    print(jaccard_score(y_test, yhat, average='micro'))

    # print(confusion_matrix(y_test, yhat, labels=[1,0]))
    print("classification_report -")
    print(classification_report(y_test, yhat))

    print(y_test[0:10])
    print(yhat[0:10])

    # print("log loss -")
    # print(log_loss(y_test, yhat))


# Model Evaluation using Test set

from sklearn.metrics import f1_score
# for f1_score please set the average parameter to 'micro'
from sklearn.metrics import log_loss

def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1


# Question 5 Calculate the F1 score and Jaccard score for each model from above.
# Use the Hyperparameter that performed best on the validation data.
# For f1_score please set the average parameter to 'micro'.

# Load Test set for evaluation

test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset/ML0101EN_EDX_skill_up/basketball_train.csv',error_bad_lines=False)
print(test_df.head())

test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
test_Feature = test_df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
test_Feature['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
test_X=test_Feature
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_X[0:5]

test_y = test_df1['POSTSEASON'].values
print(test_y[0:5])

# KNN

# Training

k = 5
# Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
print(neigh)

# Predicting

yhat = neigh.predict(test_X)
print(yhat[0:5])
print(test_y[0:5])
print(yhat.shape)
print(test_y.shape)

print("f1_score: ")
print(f1_score(test_y, yhat, average='micro'))

print("jaccard score: ")
print(jaccard_index(yhat, test_y))

# Decision Tree

# Modeling

k = 4
dTree = DecisionTreeClassifier(criterion='entropy', max_depth=k)
print(dTree)

dTree.fit(X_train, y_train)

# Prediction

yhat = dTree.predict(test_X)

print("f1_score: ")
print(f1_score(test_y, yhat, average='micro'))

print("jaccard score: ")
print(jaccard_index(yhat, test_y))


# SVM

kernel = 'rbf'
clf = svm.SVC(kernel=kernel)
clf.fit(X_train, y_train)

print(clf)

# After being fitted, the model can then be used to predict new values:
yhat = clf.predict(test_X)

print("f1_score: ")
print(f1_score(test_y, yhat, average='micro'))

print("jaccard score: ")
print(jaccard_index(yhat, test_y))

# Logistic Regression

solver='sag'
LR = LogisticRegression(C=0.01, solver=solver).fit(X_train, y_train)

yhat = LR.predict(test_X)

print("f1_score: ")
print(f1_score(test_y, yhat, average='micro'))

print("jaccard score: ")
print(jaccard_index(yhat, test_y))























