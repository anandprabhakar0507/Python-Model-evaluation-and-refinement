
.. raw:: html

   <div class="alert alert-block alert-info" style="margin-top: 20px">

.. raw:: html

   <h1 align="center">

 Link

.. raw:: html

   </h1>

.. raw:: html

   <h1 align="center">

Data Analysis with Python

.. raw:: html

   </h1>

Module 5: Model Evaluation and Refinement
=========================================

We have built models and made predictions of vehicle prices. Now we will
determine how accurate these predictions are.

Table of contents
=================

.. raw:: html

   <p>

.. raw:: html

   </p>

.. raw:: html

   <li>

Model Evaluation

.. raw:: html

   </li>

.. raw:: html

   <li>

Over-fitting, Under-fitting and Model Selection

.. raw:: html

   </li>

.. raw:: html

   <li>

Ridge Regression

.. raw:: html

   </li>

.. raw:: html

   <li>

Grid Search

.. raw:: html

   </li>

.. raw:: html

   <p>

.. raw:: html

   </p>

.. code:: python

    import pandas as pd
    import numpy as np
    
    # Import clean data 
    path = path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
    df = pd.read_csv(path)

First let's only use numeric data:

.. code:: python

    df=df._get_numeric_data()

Libraries for plotting:

.. code:: python

    from IPython.display import display
    from IPython.html import widgets 
    from IPython.display import display
    from ipywidgets import interact, interactive, fixed, interact_manual
    print("done")

Functions for plotting
----------------------

.. code:: python

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
        

.. code:: python

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


Part 1: Training and Testing
============================

An important step in testing your model is to split your data into
training and testing data. We will place the target data **price** in a
separate dataframe **y**:

.. code:: python

    y_data=df['price']

Drop price data in x data:

.. code:: python

    x_data=df.drop('price',axis=1)

Now we randomly split our data into training and testing data using the
function **train\_test\_split**:

.. code:: python

    from sklearn.model_selection import train_test_split
    
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
    
    
    print("number of test samples :", x_test.shape[0])
    print("number of training samples:",x_train.shape[0])


The **test\_size** parameter sets the proportion of data that is split
into the testing set. In the above, the testing set is set to 10% of the
total dataset.

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #1:

.. raw:: html

   </h1>

 Use the function "train\_test\_split" to split up the data set such
that 40% of the data samples will be utilized for testing, and set the
parameter "random\_state" equal to zero. The output of the function
should be the following: "x\_train\_1" , "x\_test\_1", "y\_train\_1" and
"y\_test\_1":

.. raw:: html

   </div>


.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q1" class="collapse">

::

    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0) 

.. raw:: html

   </div>

Let's import **LinearRegression** from the module **linear\_model**:

.. code:: python

    from sklearn.linear_model import LinearRegression

We create a Linear Regression object:

.. code:: python

    lre=LinearRegression()

We fit the model using the feature 'horsepower':

.. code:: python

    lre.fit(x_train[['horsepower']],y_train)

Let's Calculate the R^2 on the test data:

.. code:: python

    lre.score(x_test[['horsepower']],y_test)

We can see the R^2 is much smaller using the test data:

.. code:: python

    lre.score(x_train[['horsepower']],y_train)

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #2:

.. raw:: html

   </h1>

 Find the R^2 on the test data using 90% of the data for training data:

.. raw:: html

   </div>


.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q2" class="collapse">

::

    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.9, random_state=0)
    lre.fit(x_train1[['horsepower']],y_train1)
    lre.score(x_test1[['horsepower']],y_test1)

.. raw:: html

   </div>

Sometimes you do not have sufficient testing data. As such, you may want
to perform Cross-validation. Let's go over several methods that you can
use for Cross-validation.

Cross-validation Score
----------------------

Let's import **model\_selection** from the module **cross\_val\_scor**:

.. code:: python

    from sklearn.model_selection import cross_val_score
    print("done")

We input the object, the feature in this case ' horsepower', the target
data (y\_data). The parameter 'cv' determines the number of folds; in
this case 4:

.. code:: python

    Rcross=cross_val_score(lre,x_data[['horsepower']], y_data,cv=4)

The default scoring is R^2; each element in the array has the average
R^2 value in the fold:

.. code:: python

    Rcross

We can calculate the average and standard deviation of our estimate:

.. code:: python

    print("The mean of the folds are", Rcross.mean(),"and the standard deviation is" ,Rcross.std())

We can use negative squared error as a score by setting the parameter
'scoring' metric to 'neg\_mean\_squared\_error':

.. code:: python

    -1*cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #3:

.. raw:: html

   </h1>

 Calculate the average R^2 using two folds, find the average R^2 for the
second fold utilizing the horsepower as a feature :

.. raw:: html

   </div>


.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q3" class="collapse">

::

    Rc=cross_val_score(lre,x_data[['horsepower']], y_data,cv=2)
    Rc[1]

.. raw:: html

   </div>

You can also use the function 'cross\_val\_predict' to predict the
output. The function splits up the data into the specified number of
folds, using one fold to get a prediction while the rest of the folds
are used as test data. First import the function:

.. code:: python

    from sklearn.model_selection import cross_val_predict

We input the object, the feature in this case **'horsepower'** , the
target data **y\_data**. The parameter 'cv' determines the number of
folds, in this case 4. We can produce an output:

.. code:: python

    yhat=cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
    yhat[0:5]

Part 2: Overfitting, Underfitting and Model Selection
=====================================================

It turns out that the test data sometimes referred to as the out of
sample data is a much better measure of how well your model performs in
the real world. One reason for this is overfitting; let's go over some
examples. It turns out these differences are more apparent in Multiple
Linear Regression and Polynomial Regression so we will explore
overfitting in that context.

Let's create Multiple linear regression objects and train the model
using **'horsepower'**, **'curb-weight'**, **'engine-size'** and
**'highway-mpg'** as features:

.. code:: python

    lr=LinearRegression()
    lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_train)

Prediction using training data:

.. code:: python

    yhat_train=lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    yhat_train[0:5]

Prediction using test data:

.. code:: python

    yhat_test=lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    yhat_test[0:5]

Let's perform some model evaluation using our training and testing data
separately. First we import the seaborn and matplotlibb library for
plotting:

.. code:: python

    import matplotlib.pyplot as plt
    %matplotlib inline
    import seaborn as sns

Let's examine the distribution of the predicted values of the training
data:

.. code:: python

    Title='Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution '
    DistributionPlot(y_train,yhat_train,"Actual Values (Train)","Predicted Values (Train)",Title)

Figure 1: Plot of predicted values using the training data compared to
the training data.

So far the model seems to be doing well in learning from the training
dataset. But what happens when the model encounters new data from the
testing dataset? When the model generates new values from the test data,
we see the distribution of the predicted values is much different from
the actual target values.

.. code:: python

    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

Figure 2: Plot of predicted value compared to the actual value using the
test data.

Comparing Figure 1 and Figure 2, it is evident that the distribution of
the test data in Figure 1 is much better at fitting the data. This
difference in Figure 2 is apparent where the ranges are from 5000 to 15
000. This is where the distribution shape is exceptionally different.
Let's see if polynomial regression also exhibits a drop in the
prediction accuracy when analysing the test dataset.

.. code:: python

    from sklearn.preprocessing import PolynomialFeatures
    print("done")

Overfitting
^^^^^^^^^^^

Overfitting occurs when the model fits the noise, not the underlying
process. Therefore when testing your model using the test-set, your
model does not perform as well as it is modelling noise, not the
underlying process that generated the relationship. Let's create a
degree 5 polynomial model.

Let's use 55 percent of the data for testing and the rest for training:

.. code:: python

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
    print("done")

We will perform a degree 5 polynomial transformation on the feature
**'horse power'**:

.. code:: python

    pr=PolynomialFeatures(degree=5)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    x_test_pr=pr.fit_transform(x_test[['horsepower']])
    pr

Now let's create a linear regression model "poly" and train it:

.. code:: python

    poly=LinearRegression()
    poly.fit(x_train_pr,y_train)

We can see the output of our model using the method "predict", then
assign the values to "yhat":

.. code:: python

    yhat=poly.predict(x_test_pr )
    yhat[0:5]

Let's take the first five predicted values and compare it to the actual
targets:

.. code:: python

    print("Predicted values:", yhat[0:4])
    print("True values:",y_test[0:4].values)

We will use the function "PollyPlot" that we defined at the beginning of
the lab to display the training data, testing data, and the predicted
function:

.. code:: python

    PollyPlot(x_train[['horsepower']],x_test[['horsepower']],y_train,y_test,poly,pr)

Figure 4: A polynomial regression model. Red dots represent training
data, green dots represent test data, and the blue line represents the
model prediction.

We see that the estimated function appears to track the data but at
around 200 horsepower, the function begins to diverge from the data
points.

R^2 of the training data:

.. code:: python

    poly.score(x_train_pr, y_train)

R^2 of the test data:

.. code:: python

    poly.score(x_test_pr, y_test)

We see the R^2 for the training data is 0.5567 while the R^2 on the test
data was -29.87. The lower the R^2, the worse the model; a Negative R^2
is a sign of overfitting.

Let's see how the R^2 changes on the test data for different order
polynomials and plot the results:

.. code:: python

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

We see the R^2 gradually increases until an order three polynomial is
used. Then the R^2 dramatically decreases at four.

The following function will be used in the next section. Please run the
cell.

.. code:: python

    def f(order,test_data):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
        pr=PolynomialFeatures(degree=order)
        x_train_pr=pr.fit_transform(x_train[['horsepower']])
        x_test_pr=pr.fit_transform(x_test[['horsepower']])
        poly=LinearRegression()
        poly.fit(x_train_pr,y_train)
        PollyPlot(x_train[['horsepower']],x_test[['horsepower']],y_train,y_test,poly,pr)


The following interface allows you to experiment with different
polynomial orders and different amounts of data.

.. code:: python

    interact(f, order=(0,6,1),test_data=(0.05,0.95,0.05))

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #4(a):

.. raw:: html

   </h1>

 We can perform polynomial transformations with more than one feature.
Create a "PolynomialFeatures" object "pr1" of degree two:

.. raw:: html

   </div>

.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q2a" class="collapse">

::

    pr1=PolynomialFeatures(degree=2)

.. raw:: html

   </div>

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #4(b):

.. raw:: html

   </h1>

 Transform the training and testing samples for the features
'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg'. Hint: use
the method "fit\_transform":

.. raw:: html

   </div>

.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q2b" class="collapse">

::

    x_train_pr1=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    x_test_pr1=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

.. raw:: html

   </div>

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #4(c):

.. raw:: html

   </h1>

 How many dimensions does the new feature have? Hint: use the attribute
"shape":

.. raw:: html

   </div>

.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q2c" class="collapse">

::

    There are now 15 features: x_train_pr1.shape 

.. raw:: html

   </div>

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #4(d):

.. raw:: html

   </h1>

 Create a linear regression model "poly1" and train the object using the
method "fit" using the polynomial features:

.. raw:: html

   </div>

.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q2d" class="collapse">

::

    poly1=linear_model.LinearRegression().fit(x_train_pr1,y_train)

.. raw:: html

   </div>

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #4e):

.. raw:: html

   </h1>

Use the method "predict" to predict an output on the polynomial
features, then use the function "DistributionPlot" to display the
distribution of the predicted output vs the test data:

.. raw:: html

   </div>

.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q2e" class="collapse">

::

    yhat_test1=poly1.predict(x_train_pr1)
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,yhat_test1,"Actual Values (Test)","Predicted Values (Test)",Title)

.. raw:: html

   </div>

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #4(f):

.. raw:: html

   </h1>

 Use the distribution plot to determine the two regions were the
predicted prices are less accurate than the actual prices:

.. raw:: html

   </div>

.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q2f" class="collapse">

::

    The predicted value is lower than actual value for cars where the price  $ 10,000 range. Conversely, the predicted price is larger than the price cost in the $30, 000 to $40,000 range. As such, the model is not as accurate in these ranges.  

.. raw:: html

   </div>

Part 3: Ridge Regression
------------------------

In this section, we will review Ridge Regression. We will see how the
parameter Alfa changes the model. Just a note here, our test data will
be used as validation data.

Let's perform a degree two polynomial transformation on our data:

.. code:: python

    pr=PolynomialFeatures(degree=2)
    x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
    x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

Let's import **Ridge** from the module **linear models**:

.. code:: python

    from sklearn.linear_model import Ridge

Let's create a Ridge regression object, setting the regularization
parameter to 0.1:

.. code:: python

    RigeModel=Ridge(alpha=0.1)

Like regular regression, you can fit the model using the method **fit**:

.. code:: python

    RigeModel.fit(x_train_pr,y_train)

Similarly, you can obtain a prediction:

.. code:: python

    yhat=RigeModel.predict(x_test_pr)

Let's compare the first five predicted samples to our test set:

.. code:: python

    print('predicted:', yhat[0:4])
    print('test set :', y_test[0:4].values)

We select the value of Alfa that minimizes the test error. For example,
we can use a for loop:

.. code:: python

    Rsqu_test=[]
    Rsqu_train=[]
    dummy1=[]
    ALFA=5000*np.array(range(0,10000))
    for alfa in ALFA:
        RigeModel=Ridge(alpha=alfa) 
        RigeModel.fit(x_train_pr,y_train)
        Rsqu_test.append(RigeModel.score(x_test_pr,y_test))
        Rsqu_train.append(RigeModel.score(x_train_pr,y_train))

We can plot out the value of R^2 for different Alphas:

.. code:: python

    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    plt.plot(ALFA,Rsqu_test,label='validation data  ')
    plt.plot(ALFA,Rsqu_train,'r',label='training Data ')
    plt.xlabel('alpha')
    plt.ylabel('R^2')
    plt.legend()


Figure 6: The blue line represents the R^2 of the test data, and the red
line represents the R^2 of the training data. The x-axis represents the
different values of Alfa.

The red line in Figure 6 represents the R^2 of the test data; as Alpha
increases the R^2 decreases. Therefore, as Alfa increases, the model
performs worse on the test data. The blue line represents the R^2 on the
validation data, as the value for Alfa increases the R^2 decreases.

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #5:

.. raw:: html

   </h1>

Perform Ridge regression and calculate the R^2 using the polynomial
features. Use the training data to train the model and test data to test
the model. The parameter alpha should be set to 10:

.. raw:: html

   </div>


.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q5" class="collapse">

::

    RigeModel=Ridge(alpha=0) 
    RigeModel.fit(x_train_pr,y_train)
    RigeModel.score(x_test_pr, y_test)

.. raw:: html

   </div>

Part 4: Grid Search
-------------------

The term Alfa is a hyperparameter. Sklearn has the class
**GridSearchCV** to make the process of finding the best hyperparameter
simpler.

Let's import **GridSearchCV** from the module **model\_selection**:

.. code:: python

    from sklearn.model_selection import GridSearchCV
    print("done")

We create a dictionary of parameter values:

.. code:: python

    parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000]}]
    parameters1

Create a ridge regions object:

.. code:: python

    RR=Ridge()
    RR

Create a ridge grid search object:

.. code:: python

    Grid1 = GridSearchCV(RR, parameters1,cv=4)

Fit the model:

.. code:: python

    Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)

The object finds the best parameter values on the validation data. We
can obtain the estimator with the best parameters and assign it to the
variable BestRR as follows:

.. code:: python

    BestRR=Grid1.best_estimator_
    BestRR

We now test our model on the test data:

.. code:: python

    BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_test)

.. raw:: html

   <div class="alert alert-danger alertdanger" style="margin-top: 20px">

.. raw:: html

   <h1>

Question #6:

.. raw:: html

   </h1>

Perform a grid search for the alpha parameter and the normalization
parameter, then find the best values of the parameters:

.. raw:: html

   </div>


.. raw:: html

   <div align="right">

Click here for the solution

.. raw:: html

   </div>

.. raw:: html

   <div id="q6" class="collapse">

::

    parameters2= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000],'normalize':[True,False]} ]
    Grid2 = GridSearchCV(Ridge(), parameters2,cv=4)
    Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)
    Grid2.best_estimator

.. raw:: html

   </div>

About the Authors:
==================

This notebook written `Joseph Santarcangelo
PhD <https://www.linkedin.com/in/joseph-s-50398b136/>`__

Copyright © 2017
`cognitiveclass.ai <cognitiveclass.ai?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu>`__.
This notebook and its source code are released under the terms of the
`MIT License <https://bigdatauniversity.com/mit-license/>`__.

.. raw:: html

   <div class="alert alert-block alert-info" style="margin-top: 20px">

.. raw:: html

   <h1 align="center">

 Link

.. raw:: html

   </h1>
