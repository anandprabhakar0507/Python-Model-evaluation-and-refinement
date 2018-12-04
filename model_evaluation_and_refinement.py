
# coding: utf-8

# <div class="alert alert-block alert-info" style="margin-top: 20px">
#  <a href="http://cocl.us/DA0101EN_NotbookLink_Top"><img src = "https://ibm.box.com/shared/static/fvp89yz8uzmr5q6bs6wnguxbf8x91z35.png" width = 750, align = "center"></a>
#  <h1 align=center><font size = 5> Link</font></h1> 

#  <a href="https://www.bigdatauniversity.com"><img src = "https://ibm.box.com/shared/static/ugcqz6ohbvff804xp84y4kqnvvk3bq1g.png" width = 300, align = "center"></a>
# 
# <h1 align=center><font size = 5>Data Analysis with Python</font></h1>

# # Module 5: Model Evaluation and Refinement 
# 
# We have built models and made predictions of vehicle prices. Now we will determine how accurate these predictions are. 
# 
# 
# 

# # Table of contents
# <p></p>
# <li><a href="#ref1">Model Evaluation </a></li>
# <li><a href="#ref2">Over-fitting, Under-fitting and Model Selection </a></li>
# <li><a href="#ref3">Ridge Regression </a></li>
# <li><a href="#ref4">Grid Search</a></li>
# <p></p>

# In[ ]:

import pandas as pd
import numpy as np

# Import clean data 
path = path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)


#  First let's only use numeric data: 

# In[ ]:

df=df._get_numeric_data()


#  Libraries for plotting: 

# In[ ]:

from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
print("done")


# ## Functions for plotting 

# In[ ]:

def DistributionPlot(RedFunction,BlueFunction,RedName,BlueName,Title ):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()
    


# In[ ]:

def PollyPlot(xtrain,xtest,y_train,y_test,lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(),xtest.values.max()])

    xmin=min([xtrain.values.min(),xtest.values.min()])

    x=np.arange(xmin,xmax,0.1)


    plt.plot(xtrain,y_train,'ro',label='Training Data')
    plt.plot(xtest,y_test,'go',label='Test Data')
    plt.plot(x,lr.predict(poly_transform.fit_transform(x.reshape(-1,1))),label='Predicted Function')
    plt.ylim([-10000,60000])
    plt.ylabel('Price')
    plt.legend()


# <a id="ref1"></a>
# 
# # Part 1: Training and Testing
# 
# An important step in testing your model is to split your data into training and testing data. We will place the target data **price** in a separate dataframe **y**:

# In[ ]:

y_data=df['price']


# Drop price data in x data:

# In[ ]:

x_data=df.drop('price',axis=1)


#  Now we randomly split our data into training and testing data  using the function **train_test_split**: 

# In[ ]:

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# The **test_size** parameter sets the proportion of data that is split into the testing set. In the above, the testing set is set to 10% of the total dataset. 

#  <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #1: </h1>
# 
# <b> Use the function "train_test_split" to split up the data set such that 40% of the data samples will be utilized for testing, and set the parameter "random_state" equal to zero. The output of the function should be the following:  "x_train_1" , "x_test_1", "y_train_1" and  "y_test_1":</b>
# </div>

# In[ ]:




#  <div align="right">
# <a href="#q1" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q1" class="collapse">
# ```
# x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0) 
# ```
# </div>
# 
# 

#  Let's import **LinearRegression** from the module **linear_model**:

# In[ ]:

from sklearn.linear_model import LinearRegression


#  We create a Linear Regression object:

# In[ ]:

lre=LinearRegression()


# We fit the model using the feature 'horsepower': 

# In[ ]:

lre.fit(x_train[['horsepower']],y_train)


# Let's Calculate the R^2 on the test data:

# In[ ]:

lre.score(x_test[['horsepower']],y_test)


# We can see the R^2 is much smaller using the test data:

# In[ ]:

lre.score(x_train[['horsepower']],y_train)


#  <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #2: </h1>
# <b> 
# Find the R^2  on the test data using 90% of the data for training data:
# </b>
# </div>

# In[ ]:




#  <div align="right">
# <a href="#q2" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q2" class="collapse">
# ```
# x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.9, random_state=0)
# lre.fit(x_train1[['horsepower']],y_train1)
# lre.score(x_test1[['horsepower']],y_test1)
# ```
# </div>
# 

#  Sometimes you do not have sufficient testing data. As such, you may want to perform Cross-validation. Let's  go over several methods that you can use for  Cross-validation. 

# ## Cross-validation Score 

# Let's import **model_selection** from the module **cross_val_scor**:

# In[ ]:

from sklearn.model_selection import cross_val_score
print("done")


# We input the object, the feature in this case ' horsepower', the target data (y_data). The parameter 'cv'  determines the number of folds; in this case 4: 

# In[ ]:

Rcross=cross_val_score(lre,x_data[['horsepower']], y_data,cv=4)


# The default scoring is R^2; each element in the array has the average  R^2 value in the fold:

# In[ ]:

Rcross


#  We can calculate the average and standard deviation of our estimate:

# In[ ]:

print("The mean of the folds are", Rcross.mean(),"and the standard deviation is" ,Rcross.std())


#  We can use negative squared error as a score by setting the parameter  'scoring' metric to 'neg_mean_squared_error': 

# In[ ]:

-1*cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')


# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #3: </h1>
# <b> 
# Calculate the average R^2 using two folds, find the average R^2 for the second fold utilizing the horsepower as a feature : 
# </b>
# </div>

# In[ ]:




# <div align="right">
# <a href="#q3" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q3" class="collapse">
# ```
# Rc=cross_val_score(lre,x_data[['horsepower']], y_data,cv=2)
# Rc[1]
# ```
# </div>
# 

# You can also use the function 'cross_val_predict' to predict the output. The function splits up the data into the specified number of folds, using one fold to get a prediction while the rest of the folds are used as test data. First import the function:

# In[ ]:

from sklearn.model_selection import cross_val_predict


#  We input the object, the feature in this case **'horsepower'** , the target data **y_data**. The parameter 'cv' determines the number of folds, in this case 4.  We can produce an output:

# In[ ]:

yhat=cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
yhat[0:5]


# 
# <a id="ref2"></a>
# 
# # Part 2: Overfitting, Underfitting and Model Selection 
# 
# It turns out that the test data sometimes referred to as the out of sample data is a much better measure of how well your model performs in the real world.  One reason for this is overfitting; let's go over some examples. It turns out these differences are more apparent in Multiple Linear Regression and Polynomial Regression so we will explore overfitting in that context.

# Let's create Multiple linear regression objects and train the model using **'horsepower'**, **'curb-weight'**, **'engine-size'** and **'highway-mpg'** as features:

# In[ ]:

lr=LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_train)


# Prediction using training data:

# In[ ]:

yhat_train=lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]


#  Prediction using test data: 

# In[ ]:

yhat_test=lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]


# Let's perform some model evaluation using our training and testing data separately. First  we import the seaborn and matplotlibb library for plotting:

# In[ ]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns


# Let's examine the distribution of the predicted values of the training data:

# In[ ]:

Title='Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution '
DistributionPlot(y_train,yhat_train,"Actual Values (Train)","Predicted Values (Train)",Title)


#  Figure 1: Plot of predicted values using the training data compared to the training data. 

# So far the model seems to be doing well in learning from the training dataset. But what happens when the model encounters new data from the testing dataset? When the model generates new values from the test data, we see the distribution of the predicted values is much different from the actual target values. 

# In[ ]:

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)


# Figure 2: Plot of predicted value compared to the actual value using the test data.

# Comparing Figure 1 and Figure 2, it is evident that the distribution of the test data in Figure 1 is much better at fitting the data. This difference in Figure 2 is apparent where the ranges are from 5000 to 15 000. This is where the distribution shape is exceptionally different. Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset. 

# In[ ]:

from sklearn.preprocessing import PolynomialFeatures
print("done")


# ####  Overfitting 
# Overfitting occurs when the model fits the noise, not the underlying process. Therefore when testing your model using the test-set, your model does not perform as well as it is modelling noise, not the underlying process that generated the relationship. Let's create a degree 5 polynomial model.

# Let's use 55 percent of the data for testing and the rest for training:

# In[ ]:

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
print("done")


# We will perform a degree 5 polynomial transformation on the feature **'horse power'**: 

# In[ ]:

pr=PolynomialFeatures(degree=5)
x_train_pr=pr.fit_transform(x_train[['horsepower']])
x_test_pr=pr.fit_transform(x_test[['horsepower']])
pr


# Now let's create a linear regression model "poly" and train it:

# In[ ]:

poly=LinearRegression()
poly.fit(x_train_pr,y_train)


#  We can see the output of our model using the method  "predict", then assign the values to "yhat":

# In[ ]:

yhat=poly.predict(x_test_pr )
yhat[0:5]


# Let's take the first five predicted values and compare it to the actual targets:

# In[ ]:

print("Predicted values:", yhat[0:4])
print("True values:",y_test[0:4].values)


# We will use the function "PollyPlot" that we defined at the beginning of the lab to display the training data, testing data, and the predicted function:

# In[ ]:

PollyPlot(x_train[['horsepower']],x_test[['horsepower']],y_train,y_test,poly,pr)


# Figure 4: A polynomial regression model. Red dots represent training data, green dots represent test data, and the blue line represents the model prediction. 

# We see that the estimated function appears to track the data but at around 200 horsepower, the function begins to diverge from the data points. 

#  R^2 of the training data:

# In[ ]:

poly.score(x_train_pr, y_train)


#  R^2 of the test data:

# In[ ]:

poly.score(x_test_pr, y_test)


# We see the R^2 for the training data is 0.5567 while the R^2 on the test data was -29.87.  The lower the R^2, the worse the model; a Negative R^2 is a sign of overfitting.

# Let's see how the R^2 changes on the test data for different order polynomials and plot the results:

# In[ ]:

Rsqu_test=[]

order=[1,2,3,4]
for n in order:
    pr=PolynomialFeatures(degree=n)
    
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr=pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr,y_train)
    
    Rsqu_test.append(lr.score(x_test_pr,y_test))

plt.plot(order,Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')    


#  We see the R^2 gradually increases until an order three polynomial is used. Then the  R^2 dramatically decreases at four.

#  The following function will be used in the next section. Please run the cell.

# In[ ]:

def f(order,test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr=PolynomialFeatures(degree=order)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    x_test_pr=pr.fit_transform(x_test[['horsepower']])
    poly=LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']],x_test[['horsepower']],y_train,y_test,poly,pr)


# The following interface allows you to experiment with different polynomial orders and different amounts of data. 

# In[ ]:

interact(f, order=(0,6,1),test_data=(0.05,0.95,0.05))


#  <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #4(a): </h1>
# 
# <b> We can perform polynomial transformations with more than one feature. Create a "PolynomialFeatures" object "pr1" of degree two:</b>
# </div>

#  <div align="right">
# <a href="#q2a" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q2a" class="collapse">
# ```
# pr1=PolynomialFeatures(degree=2)
# ```
# </div>
# 
# 

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #4(b): </h1>
# 
# <b> 
#  Transform the training and testing samples for the features 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg'. Hint: use the method "fit_transform": </b>
# </div>

#  <div align="right">
# <a href="#q2b" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q2b" class="collapse">
# ```
# x_train_pr1=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# x_test_pr1=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# ```
# </div>
# 
# 

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #4(c): </h1>
# 
# <b> 
# How many dimensions does the new feature have? Hint: use the attribute "shape":
# </b>
# </div>
# 

# <div align="right">
# <a href="#q2c" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q2c" class="collapse">
# ```
# There are now 15 features: x_train_pr1.shape 
# ```
# </div>
# 

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #4(d): </h1>
# 
# <b> 
# Create a linear regression model "poly1" and train the object using the method "fit" using the polynomial features: </b>
# </div>
# 

# <div align="right">
# <a href="#q2d" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q2d" class="collapse">
# ```
# poly1=linear_model.LinearRegression().fit(x_train_pr1,y_train)
# ```
# </div>
# 
# 

#  <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #4e): </h1>
# 
# <b> 
#  
#  Use the method  "predict" to predict an output on the polynomial features, then use the function "DistributionPlot"  to display the distribution of the predicted output vs the test data:</b>
# </div>

#  <div align="right">
# <a href="#q2e" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q2e" class="collapse">
# ```
# yhat_test1=poly1.predict(x_train_pr1)
# Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
# DistributionPlot(y_test,yhat_test1,"Actual Values (Test)","Predicted Values (Test)",Title)
# ```
# </div>
# 

#  <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #4(f): </h1>
# 
# <b> 
#  Use the distribution plot to determine the two regions were the predicted prices are less accurate than the actual prices:
#  </div>

#  <div align="right">
# <a href="#q2f" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q2f" class="collapse">
# ```
# The predicted value is lower than actual value for cars where the price  $ 10,000 range. Conversely, the predicted price is larger than the price cost in the $30, 000 to $40,000 range. As such, the model is not as accurate in these ranges.  
# ```
# <img src = "https://ibm.box.com/shared/static/c35ipv9zeanu7ynsnppb8gjo2re5ugeg.png" width = 700, align = "center">
# 
# 
# 
# </div>
# 

# <a id="ref3"></a>
# 
# ## Part 3: Ridge Regression 

#  In this section, we will review Ridge Regression. We will see how the parameter Alfa changes the model. Just a note here, our test data will be used as validation data.

#  Let's perform a degree two polynomial transformation on our data:

# In[ ]:

pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])


#  Let's import  **Ridge**  from the module **linear models**:

# In[ ]:

from sklearn.linear_model import Ridge


# Let's create a Ridge regression object, setting the regularization parameter to 0.1: 

# In[ ]:

RigeModel=Ridge(alpha=0.1)


#  Like regular regression, you can fit the model using the method **fit**:

# In[ ]:

RigeModel.fit(x_train_pr,y_train)


#  Similarly, you can obtain a prediction: 

# In[ ]:

yhat=RigeModel.predict(x_test_pr)


# Let's compare the first five predicted samples to our test set: 

# In[ ]:

print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)


#  We select the value of Alfa that minimizes the test error. For example, we can use a for loop: 

# In[ ]:

Rsqu_test=[]
Rsqu_train=[]
dummy1=[]
ALFA=5000*np.array(range(0,10000))
for alfa in ALFA:
    RigeModel=Ridge(alpha=alfa) 
    RigeModel.fit(x_train_pr,y_train)
    Rsqu_test.append(RigeModel.score(x_test_pr,y_test))
    Rsqu_train.append(RigeModel.score(x_train_pr,y_train))


# We can plot out the value of R^2 for different Alphas: 

# In[ ]:

width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALFA,Rsqu_test,label='validation data  ')
plt.plot(ALFA,Rsqu_train,'r',label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()


#   Figure 6: The blue line represents the R^2 of the test data, and the red line represents the R^2 of the training data. The x-axis represents the different values of Alfa. 

#  The red line in Figure 6 represents the  R^2 of the test data; as Alpha increases the R^2 decreases. Therefore, as Alfa increases, the model performs worse on the test data.  The blue line represents the R^2 on the validation data, as the value for Alfa increases the R^2 decreases.   

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #5: </h1>
# 
# Perform Ridge regression and calculate the R^2 using the polynomial features. Use the training data to train the model and test data to test the model. The parameter alpha should be set to  10:
# </div>

# In[ ]:




# <div align="right">
# <a href="#q5" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q5" class="collapse">
# ```
# RigeModel=Ridge(alpha=0) 
# RigeModel.fit(x_train_pr,y_train)
# RigeModel.score(x_test_pr, y_test)
# ```
# </div>
# 

# <a id="ref4"></a>
# 
# ## Part 4: Grid Search

# The term Alfa is a hyperparameter. Sklearn has the class  **GridSearchCV** to make the process of finding the best hyperparameter simpler.

#  Let's import **GridSearchCV** from  the module **model_selection**:

# In[ ]:

from sklearn.model_selection import GridSearchCV
print("done")


# We create a dictionary of parameter values:

# In[ ]:

parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000]}]
parameters1


# Create a ridge regions object:

# In[ ]:

RR=Ridge()
RR


# Create a ridge grid search object: 

# In[ ]:

Grid1 = GridSearchCV(RR, parameters1,cv=4)


# Fit the model: 

# In[ ]:

Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)


# The object finds the best parameter values on the validation data. We can obtain the estimator with the best parameters and assign it to the variable BestRR as follows:

# In[ ]:

BestRR=Grid1.best_estimator_
BestRR


#  We now test our model on the test data: 

# In[ ]:

BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_test)


# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> Question  #6: </h1>
# Perform a grid search for the alpha parameter and the normalization parameter, then find the best values of the parameters:
# 
# </div>

# In[ ]:




#  <div align="right">
# <a href="#q6" class="btn btn-default" data-toggle="collapse">Click here for the solution</a>
# 
# </div>
# <div id="q6" class="collapse">
# ```
# parameters2= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000],'normalize':[True,False]} ]
# Grid2 = GridSearchCV(Ridge(), parameters2,cv=4)
# Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)
# Grid2.best_estimator
# ```
# </div>
# 

# # About the Authors:  
# 
# This notebook written [Joseph Santarcangelo PhD]( https://www.linkedin.com/in/joseph-s-50398b136/)

# Copyright &copy; 2017 [cognitiveclass.ai](cognitiveclass.ai?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
# 

#  <div class="alert alert-block alert-info" style="margin-top: 20px">
#  <a href="http://cocl.us/DA0101EN_NotbookLink_bottom"><img src = "https://ibm.box.com/shared/static/cy2mwm7519t4z6dxefjpzgtbpi9p8l7h.png" width = 750, align = "center"></a>
#  <h1 align=center><font size = 5> Link</font></h1> 
