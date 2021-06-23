# loan-approval-prediction
Project on Loan Prediction Analysis

### Learning points:

**Tackling null values in columns**

* fill null values in numeric columns with *mean* 

* fill null values in categorical columns with *mode*


**Log Transformation**

* used to transform highly skewed variabls into more normalized dataset

* transforming model(s) to take advantage of statistical tools such as linear regression that improve on features that are normally distributed

> A regression model will have unit changes between the x and y variables, where a single unit change in x will coincide with a constant change in y. Taking the log of one or both variables will effectively change the case from a unit change to a percent change. This is especially important when using medium to large datasets. A logarithm is the base of a positive number. For example, the base10 log of 100 is 2, because 10^2 = 100. So the natural log function and the exponential function (ex) are inverses of each other.

Explanation reference: https://dev.to/rokaandy/logarithmic-transformation-in-linear-regression-models-why-when-3a7c

**Label Encoding**
* make use of sklearn LabelEncoder to transform categorical data to numeric data e.g. male, female to 0,1

**Cross Validation Score**
* Tackles the overfitting problem

Explanation reference: https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85

**Random Forest VS Extra Trees**
* The main difference between random forests and extra trees (usually called extreme random forests) lies in the fact that, instead of computing the locally optimal feature/split combination (for the random forest), for each feature under consideration, a random value is selected for the split (for the extra trees)
* In terms of computational cost, and therefore execution time, the Extra Trees algorithm is faster. This algorithm saves time because the whole procedure is the same, but it randomly chooses the split point and does not calculate the optimal one.

**XGBoost**
* stands for eXtreme Gradient Boosting

Explanation reference: https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/

**LightGBM**
* short for Light Gradient Boosting Machine

Documentation: https://lightgbm.readthedocs.io/en/latest/

**catboost**
* a machine learning algorithm that uses gradient boosting on decision trees

Documentation: https://catboost.ai/docs

**XGBoost VS LightGBM VS catboost**

<img width="800" alt="kaggle" src="https://user-images.githubusercontent.com/57902840/123130803-ab9c1480-d47f-11eb-9df2-5caef8d351d7.PNG">

Creds to Kaggle

**Hyperparameter tuning - RandomizedSearchCV**
* Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the built model. It is similar to grid search, and yet it has proven to yield better results comparatively. The drawback of random search is that it yields high variance during computing.

Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html


