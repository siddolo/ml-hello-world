# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# LOAD THE DATASET
#file = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
file = './iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(file, names=names)


# SUMMARIZE THE DATASET

#shape
"""
We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
"""
print('\n\n==== Dimension of Dataset')
print(dataset.shape)

# head
"""
It is also always a good idea to actually eyeball your data.
You should see the first 20 rows of the data:
"""
print('\n\n==== Peek at the Data')
print(dataset.head(20))

# descriptions
"""
Now we can take a look at a summary of each attribute.
This includes the count, mean, the min and max values as well as some percentiles.
We can see that all of the numerical values have the same scale (centimeters) and similar ranges between 0 and 8 centimeters.

"""
print('\n\n==== Statistical Summary')
print(dataset.describe())

# class distribution
"""
Let's now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.
"""
print('\n\n==== Class Distribution')
print(dataset.groupby('class').size())

# DATA VISUALIZATION

# Univariate Plots
"""
We start with some univariate plots, that is, plots of each individual variable.
Given that the input variables are numeric, we can create box and whisker plots of each.
This gives us a much clearer idea of the distribution of the input attributes:
"""
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

"""
We can also create a histogram of each input variable to get an idea of the distribution.
It looks like perhaps two of the input variables have a Gaussian distribution.
This is useful to note as we can use algorithms that can exploit this assumption.
"""
# histograms
dataset.hist()
plt.show()

# Multivariate Plots
"""
Now we can look at the interactions between the variables.
First, let's look at scatterplots of all pairs of attributes.
This can be helpful to spot structured relationships between input variables.

Note the diagonal grouping of some pairs of attributes.
This suggests a high correlation and a predictable relationship.
"""

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# EVALUATE SOME ALGORITHMS
"""
Now it is time to create some models of the data and estimate their accuracy on unseen data.
Here is what we are going to cover in this step:

- Separate out a validation dataset.
- Set-up the test harness to use 10-fold cross validation.
- Build 5 different models to predict species from flower measurements
- Select the best model.

# Create a Validation Dataset
We need to know that the model we created is any good.
Later, we will use statistical methods to estimate the accuracy of the models that we create on unseen data.
We also want a more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual unseen data.

That is, we are going to hold back some data that the algorithms will not get to see and we will use this data
to get a second and independent idea of how accurate the best model might actually be.

We will split the loaded dataset into two, 80% of which we will use to train our models and 20%
that we will hold back as a validation dataset.
"""
# Split-out validation dataset
"""
You now have training data in the X_train and Y_train for preparing models and a X_validation and Y_validation sets that we can use later.
Notice that we used a python slice to select the columns in the NumPy array. If this is new to you, you might want to check-out this post:
https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
"""
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test Harness
"""
We will use 10-fold cross validation to estimate accuracy.
This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.
"""

# Test options and evaluation metric
"""
The specific random seed does not matter, learn more about pseudorandom number generators here:
https://machinelearningmastery.com/introduction-to-random-number-generators-for-machine-learning/
We are using the metric of 'accuracy' to evaluate models. This is a ratio of the number of correctly
predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give
a percentage (e.g. 95% accurate).
We will be using the scoring variable when we run build and evaluate each model next.
"""
seed = 7
scoring = 'accuracy'

# Build Models
"""
We don't know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results.

Let's evaluate 6 different algorithms:

- Logistic Regression (LR)
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN).
- Classification and Regression Trees (CART).
- Gaussian Naive Bayes (NB).
- Support Vector Machines (SVM).
- This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.

Let's build and evaluate our models:
"""
# Spot Check Algorithms
print('\n\n==== Models')
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

"""
Select Best Model
We now have 6 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate.

Running the example above, we get the following raw results:

LR: 0.966667 (0.040825)
LDA: 0.975000 (0.038188)
KNN: 0.983333 (0.033333)
CART: 0.975000 (0.038188)
NB: 0.975000 (0.053359)
SVM: 0.991667 (0.025000)

Note, you're results may differ. For more on this see the post: https://machinelearningmastery.com/randomness-in-machine-learning/

In this case, we can see that it looks like Support Vector Machines (SVM) has the largest estimated accuracy score.

We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model.
There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
"""

# Compare Algorithms
"""
You can see that the box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy.
"""
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# MAKE PREDICTIONS
"""
The KNN algorithm is very simple and was an accurate model based on our tests.
Now we want to get an idea of the accuracy of the model on our validation set.

This will give us an independent final check on the accuracy of the best model.
It is valuable to keep a validation set just in case you made a slip during training,
such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.

We can run the KNN model directly on the validation set and summarize the results
as a final accuracy score, a confusion matrix and a classification report.
"""

# Make predictions on validation dataset
"""
We can see that the accuracy is 0.9 or 90%.
The confusion matrix provides an indication of the three errors made.
Finally, the classification report provides a breakdown of each class by precision,
recall, f1-score and support showing excellent results (granted the validation dataset was small).
You can learn more about how to make predictions and predict probabilities here:

How to Make Predictions with scikit-learn: https://machinelearningmastery.com/make-predictions-scikit-learn/
"""
print('\n\n==== Make predictions on validation dataset')
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
