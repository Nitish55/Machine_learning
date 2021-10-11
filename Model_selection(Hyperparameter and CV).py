#!/usr/bin/env python
# coding: utf-8

# ## Cross-Validation with Linear Regression
# 
# This notebook demonstrates how to do cross-validation (CV) with linear regression as an example (it is heavily used in almost all modelling techniques such as decision trees, SVM etc.). We will mainly use `sklearn` to do cross-validation.
# 
# This notebook is divided into the following parts:
# 0. Experiments to understand overfitting
# 1. Building a linear regression model without cross-validation
# 2. Problems in the current approach
# 3. Cross-validation: A quick recap
# 4. Cross-validation in `sklearn`:`
#     - 4.1 K-fold CV 
#     - 4.2 Hyperparameter tuning using CV
#     - 4.3 Other CV schemes

# ## 0. Experiments to Understand Overfitting
# 
# In this section, let's quickly go through some experiments to understand what overfitting looks like. We'll run some experiments using polynomial regression.

# In[1]:


# import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler


# ## Importing the data

# In[2]:


df1=pd.read_csv("Housing.csv")
df1.head()


# In[3]:


df1.info()


# In[4]:


df1.isnull().sum()


# In[5]:


df1.shape


# #### For the first experiment, we'll do regression with only one feature. Let's filter the data so it only contains area and price.

# In[6]:


df=df1[["area","price"]]
df.head()


# ## Rescaling the features
# As `area` and `price` are far apart we need rescaling, as we studied in scaling topic

# In[7]:


df_cols=df.columns
scaler=MinMaxScaler()

#Fit te columns
df=scaler.fit_transform(df)
df


# In[8]:


# rename columns (since now its an np array)
df=pd.DataFrame(df,columns=df_cols)    
#df.columns=df_cols
df.head()


# ## Visulaize the realtion between area and price

# In[9]:


# visualise area-price relationship
sns.regplot(x="area",y="price",data=df)


# ## Splitting the data into train_test split

# In[10]:


df_train,df_test=train_test_split(df,test_size=0.3,random_state=10)
print(len(df_train))
print(len(df_test))


# In[11]:


# split into X and y for both train and test sets
# reshaping is required since sklearn requires the data to be in shape (n, 1), not as a series of shape (n, )
X_train=df_train["area"]
X_train.shape


# In[12]:


X_train=X_train.values.reshape(-1,1)
y_train=df_train["price"]


# Similarly for X_test
X_test=df_test["area"]
X_test=X_test.values.reshape(-1,1)
y_test=df_test["price"]


# In[13]:


X_train.shape


# In[14]:


y_train.shape


# ### Polynomial Regression
# 
# You already know simple linear regression:
# 
# $y = \beta_0 + \beta_1 x_1$
# 
# In polynomial regression of degree $n$, we fit a curve of the form:
# 
# $y = \beta_0 + \beta_1 x_1 + \beta_2x_1^2 + \beta_3x_1^3 ... + \beta_nx_1^n$
# 
# In the experiment below, we have fitted polynomials of various degrees on the housing data and compared their performance on train and test sets.
# 
# In sklearn, polynomial features can be generated using the `PolynomialFeatures` class. Also, to perform `LinearRegression` and `PolynomialFeatures` in tandem, we will use the module `sklearn_pipeline` - it basically creates the features and feeds the output to the model (in that sequence).

# In[15]:


len(X_train)


# Let's now predict the y labels (for both train and test sets) and store the predictions in a table. Each row of the table is one data point, each column is a value of $n$ (degree).

# <table style="width:100%">
#   <tr>
#     <th>   </th>
#     <th>degree-1</th>
#     <th>degree-2</th> 
#     <th>degree-3</th>
#     <th>...</th>
#     <th>degree-n</th>
#   </tr>
#   <tr>
#     <th>x1</th>
#   </tr>
#   <tr>
#     <th>x2</th>
#   </tr>
#    <tr>
#     <th>x3</th>
#     </tr>
#     <tr>
#     <th>...</th>
#     </tr>
#     <tr>
#     <th>xn</th>
#     </tr>
# </table>

# In[16]:


# fit multiple polynomial features
degrees=[1, 2, 3, 6, 10, 20]

# initialise y_train_pred and y_test_pred matrices to store the train and test predictions
# each row is a data point, each column a prediction using a polynomial of some degree
y_train_pred= np.zeros((len(X_train),len(degrees)))
y_test_pred=np.zeros((len(X_test),len(degrees)))
y_train_pred


# ### Create pipeline to fit various degrees

# In[17]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[18]:


for i,degree in enumerate(degrees):
    # make pipeline: create features, then feed them to linear_reg model
    
    
     # make pipeline: create features, then feed them to linear_reg model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    # predict on test and train data
    # store the predictions of each degree in the corresponding column
    y_train_pred[:, i] = model.predict(X_train)
    y_test_pred[:, i] = model.predict(X_test)
  


# In[19]:


pd.DataFrame(y_train_pred,columns=degrees).head()


# ### visualise train and test predictions

# In[20]:


# note that the y axis is on a log scale

plt.figure(figsize=(16, 8))

# train data
plt.subplot(121)
plt.scatter(X_train, y_train)
plt.yscale('log')
plt.title("Train data")
for i, degree in enumerate(degrees):    
    plt.scatter(X_train, y_train_pred[:, i], s=15, label=str(degree))
    plt.legend(loc='upper left')
    
   ## test data
plt.subplot(122)
plt.scatter(X_test, y_test)
plt.yscale('log')
plt.title("Test data")
for i, degree in enumerate(degrees):    
    plt.scatter(X_test, y_test_pred[:, i], label=str(degree))
    plt.legend(loc='upper left')


# ## compare r2 for train and test sets (for all polynomial fits)

# In[21]:


print("R-squared values: \n")
for i,degree in enumerate(degrees):
    train_r2=round(sklearn.metrics.r2_score(y_train,y_train_pred[:,i]),2)
    test_r2=round(sklearn.metrics.r2_score(y_test,y_test_pred[:,i]),2)
    print("Polynomial degree {0}: train score={1}, test score={2}".format(degree, 
                                                                         train_r2, 
                                                                         test_r2))


# ## Notice
#  - With the increse in polynomial degree test score keeps on decreasing depicts with increse in polynomial degree it cause `Overfit`

# ## 1. Building a Model Without Cross-Validation
# 
# Let's now build a multiple regression model. First, let's build a vanilla MLR model without any cross-validation etc.

# In[22]:


# data preparation

# list of all the "yes-no" binary categorical variables
# we'll map yes to 1 and no to 0
binary_vars_list =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# defining the map function
def binary_map(x):
    return x.map(lambda x: 1 if x=="yes" else 0)

# applying the function to the housing variables list
df1[binary_vars_list] = df1[binary_vars_list].apply(binary_map)
df1.head()


# # Alternate Way

# In[23]:


# var_list=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# for i in df1[var_list]:
#     df1[i]=np.where(df1[i]=="yes",1,0)
# df1


# ## creating dummies variables

# In[24]:


# 'dummy' variables
# get dummy variables for 'furnishingstatus' 
# also, drop the first column of the resulting df (since n-1 dummy vars suffice)
status=pd.get_dummies(df1["furnishingstatus"],drop_first=True)
status.head()


# In[25]:


# concat the dummy variable df with the main df
df1=pd.concat([df1,status],axis=1)
df1.head()


# In[26]:


# Drop furnishingstatus
df1.drop("furnishingstatus",axis=1,inplace=True)
df1.head()


# ## Splitting into train-test

# In[27]:


len(df1["guestroom"].unique())


# ## Scaling

# In[28]:


df_train,df_test=train_test_split(df1,train_size=0.7,random_state=100)
# rescale the features
scaler = MinMaxScaler()

# apply scaler() to all the numeric columns 
#numeric_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
numeric_vars=[i for i in df1.columns if (len(df1[i].unique())>2)]
df_train[numeric_vars] = scaler.fit_transform(df_train[numeric_vars])
df_train.head()


# In[29]:


# apply rescaling to the test set also
df_test[numeric_vars] = scaler.transform(df_test[numeric_vars])
df_test.head()


# ### Divide into X_train, y_train, X_test, y_test

# In[30]:


y_train = df_train.pop('price')
X_train = df_train

y_test = df_test.pop('price')
X_test = df_test


# Note that we haven't rescaled the test set yet, which we'll need to do later while making predictions.

# #### Using RFE 
# 
# Now, we have 13 predictor features. To build the model using RFE, we need to tell RFE how many features we want in the final model. It then runs a feature elimination algorithm. 
# 
# Note that the number of features to be used in the model is a **hyperparameter**.

# In[31]:


from sklearn.feature_selection import RFE


# In[32]:


# first model with an arbitrary choice of n_features
# running RFE with number of features=10
lm=LinearRegression()
lm.fit(X_train, y_train)

rfe=RFE(lm,n_features_to_select=10)
rfe=rfe.fit(X_train,y_train)


# In[33]:


# tuples of (feature name, whether selected, ranking)
# note that the 'rank' is > 1 for non-selected features
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# ## Make predictions

# In[34]:


# predict prices of X_test
y_pred=lm.predict(X_test)
                 
# evaluate the model on test set)
r2=sklearn.metrics.r2_score(y_test,y_pred)
print(r2)


# ### Try with another value of RFE

# In[35]:


lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=6)             
rfe = rfe.fit(X_train, y_train)

# predict prices of X_test
y_pred = rfe.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


# ### NOTE-
# - Generally we test on train test before testing the test set data, but in here we check on test feature first by selecting 10 feature and again fed with 6 feature to RFE which is actuallt cheating as we are makingthe model to sneap peek into our test data

# ## 2. Problems in the Current Approach
# 
# In train-test split, we have three options:
# 1. **Simply split into train and test**: But that way tuning a hyperparameter makes the model 'see' the test data (i.e. knowledge of test data leaks into the model)
# 2. **Split into train, validation, test sets**: Then the validation data would eat into the training set
# 3. **Cross-validation**: Split into train and test, and train multiple models by sampling the train set. Finally, just test once on the test set.

# ## 3. Cross-Validation: A Quick Recap
# 
# The following figure illustrates k-fold cross-validation with k=4. There are some other schemes to divide the training set, we'll look at them briefly later.
# 
# <img src="cv.png" title="K-Fold Cross Validation" />

# ## 4. Cross-Validation in sklearn
# 
# Let's now experiment with k-fold CV.

# In[36]:


X_train.shape


# In[37]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# In[38]:


lm=LinearRegression()
score=cross_val_score(lm,X_train,y_train,cv=5,scoring="r2")
score


# ### Note-
# This score is the score is the score of held out set for eg- in cv1- one set is hed out as test and reamaing is train like that-- like in yellow filled rectange in fig above

# ## the other way of doing the same thing (more explicit)

# In[39]:


# create a KFold object with 5 splits 
folds =KFold(n_splits=5,shuffle=True,random_state=100)
scores=cross_val_score(lm,X_train,y_train,cv=folds,scoring="r2")
scores


# In[40]:


# can tune other metrics, such as MSE
scores = cross_val_score(lm, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
scores


# #### here negative value means postive of mean_sqaure error, we getting negative because we used "neg_mean_squared_error"

# ### 4.2 Hyperparameter Tuning Using Grid Search Cross-Validation (to find out suitable numbers of variables)
# 
# A common use of cross-validation is for tuning hyperparameters of a model. The most common technique is what is called **grid search** cross-validation.
# 
# 
# <img src="grid_search_image.png"/>

# In[41]:


# number of features in X_train
len(X_train.columns)


# In[43]:


# step-1: create a cross-validation scheme
folds = KFold(n_splits=5,shuffle=True,random_state=100)

# step-2: specify range of hyperparameters to tune
hyper_params =[{"n_features_to_select":list(range(1,len(X_train.columns)+1))}]


# In[44]:


hyper_params


# In[50]:


# step-3: perform grid search
# 3.1 specify model
lm=LinearRegression()
lm.fit(X_train,y_train)
rfe=RFE(lm)

# 3.2 call GridSearchCV()
model_cv=GridSearchCV(estimator=rfe,param_grid=hyper_params,scoring="r2",cv=folds,verbose=1,return_train_score=True)
# fit the model
model_cv.fit(X_train,y_train)


# ### esitmator- 
# - It takes a model in this case its rfe i.e rfe rom linear regression
# 
# #### verbose : integer it Controls the verbosity: the higher, the more messages.
# - >1 : the computation time for each fold and parameter candidate is displayed;
# - >2 : the score is also displayed;
# - >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
# #### return_train_score
# - It return both the r2 score of `train` as well as `test`, so that we can identify if there is `overfitting` or not.

# ## CV result

# In[51]:


cv_result=pd.DataFrame(model_cv.cv_results_)
cv_result


# # Note-
# - Here we are interseted in `mean_train_score` and `mean_test_score`

# In[55]:


# lets plot to find out the suitable numbers of hyperparametes i.e variables
plt.figure(figsize=(16,6))
plt.plot(cv_result["param_n_features_to_select"],cv_result["mean_train_score"])
plt.plot(cv_result["param_n_features_to_select"],cv_result["mean_test_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')


# ## Note
# - In here the optimal numbers of hyperparameter is 10 after that the curve of both train and test flattens.

# Now we can choose the optimal value of number of features and build a final model.

# In[57]:


# final model
n_features_optimal = 10

lm=LinearRegression()
lm.fit(X_train,y_train)

rfe=RFE(estimator=lm,n_features_to_select=n_features_optimal)
rfe.fit(X_train,y_train)
# predict prices of X_test
y_pred=lm.predict(X_test)
r2=sklearn.metrics.r2_score(y_test,y_pred)
print(r2)


# Notice that the test score is very close to the 'mean test score' on the k-folds (about 60%). In general, the mean score estimated by CV will usually be a good estimate of the test score.

# # Another Example: Car Price Prediction

# In[62]:


# reading the dataset
cars = pd.read_csv("CarPrice_Assignment.csv")
cars.head()


# In[63]:


# All data preparation steps in this cell

# converting symboling to categorical
cars['symboling'] = cars['symboling'].astype('object')

import re
from sklearn.preprocessing import scale
# create new column: car_company
p = re.compile(r'\w+-?\w+')
cars['car_company'] = cars['CarName'].apply(lambda x: re.findall(p, x)[0])


# replacing misspelled car_company names
# volkswagen
cars.loc[(cars['car_company'] == "vw") | 
         (cars['car_company'] == "vokswagen")
         , 'car_company'] = 'volkswagen'
# porsche
cars.loc[cars['car_company'] == "porcshce", 'car_company'] = 'porsche'
# toyota
cars.loc[cars['car_company'] == "toyouta", 'car_company'] = 'toyota'
# nissan
cars.loc[cars['car_company'] == "Nissan", 'car_company'] = 'nissan'
# mazda
cars.loc[cars['car_company'] == "maxda", 'car_company'] = 'mazda'


# drop carname variable
cars = cars.drop('CarName', axis=1)


# split into X and y
X = cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'car_company']]
y = cars['price']


# creating dummy variables for categorical variables
cars_categorical = X.select_dtypes(include=['object'])
cars_categorical.head()


# convert into dummies
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
cars_dummies.head()


# drop categorical variables 
X = X.drop(list(cars_categorical.columns), axis=1)


# concat dummy variables with X
X = pd.concat([X, cars_dummies], axis=1)


# rescale the features
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=40)


# In[64]:


# number of features
len(X_train.columns)


# In[65]:


# creating a KFold object with 5 splits 
folds = KFold(n_splits=5,shuffle=True,random_state=100)

# specify range of hyperparameters from 2 to 40 as on 1 its carID also we look for 38 features as it is more than enough
hyper_params = [{'n_features_to_select': list(range(2, 40))}]

# specify model
lm=LinearRegression()
lm.fit(X_train,y_train)
rfe=RFE(lm)
# set up GridSearchCV()
model_cv=GridSearchCV(rfe,param_grid=hyper_params,scoring="r2",cv=folds,verbose=1,return_train_score=True)
# Fit the model
model_cv.fit(X_train,y_train)


# In[68]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[70]:


# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')


# ### Final model for Car pricing

# In[71]:


# final model
n_features_optimal = 10

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(X_train, y_train)

# predict prices of X_test
y_pred = lm.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


# ### 4.3 Types of Cross-Validation Schemes
# 
# 
# 1. **K-Fold** cross-validation: Most common
# 2. **Leave One Out (LOO)**: Takes each data point as the 'test sample' once, and trains the model on the rest n-1 data points. Thus, it trains n total models.
#     - Advantage: Utilises the data well since each model is trained on n-1 samples
#     - Disadvantage: Computationally expensive
# 3. **Leave P-Out (LPO)**: Creat all possible splits after leaving p samples out. For n data points, there are (nCp) possibile train-test splits.
# 4. (**For classification problems**) **Stratified K-Fold**: Ensures that the relative class proportion is approximately preserved in each train and validation fold. Important when ther eis huge class imbalance (e.g. 98% good customers, 2% bad).

# In[ ]:




