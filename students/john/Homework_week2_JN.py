
# coding: utf-8

# In[1]:


# import all the libraries requeired
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# this allows plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Export and read the data 
mpg_data = pd.read_csv("/home/dat11_john/Homework/Auto_mpg_data_set.csv")


# In[75]:


# Verify what the data looks like 
mpg_data.head(30)


# In[6]:


# check for Null values 
mpg_data.isnull().sum()


# In[7]:


# Summerize and look at data 
mpg_data.describe()


# In[8]:


# visualize all variables to see which have a relationship
mpg_data.corr()


# In[10]:


# Import seaborn so we can create a heatmap for correlation
import seaborn as sns

mpg_data_corr = mpg_data.corr()


# In[14]:


# the above table as a heatmap to easily identify relationships
sns.heatmap(mpg_data_corr, 
        xticklabels=mpg_data_corr.columns,
        yticklabels=mpg_data_corr.columns, annot=True)


# In[15]:


# Plot each variable against each other 
# scroll down past the subplot information
# Given we're also predicting mpg, the strongest relationsphip are with cylinder, displacement, horsepower and weight
pd.scatter_matrix(mpg_data, figsize=(15,15))


# In[33]:


# Simple Linear Regression
# Create fitted models to see which variable provides a stronger relationship 
# create a fitted model in one line
mpg_displacement = smf.ols(formula='mpg ~ displacement', data=mpg_data).fit()
mpg_displacement.summary()


# In[34]:


# check the distribution of mpg vs. cylinders using a scatterplot
sns.set_style("darkgrid")

sns.lmplot(y='mpg', x='displacement', data=mpg_data)


# In[40]:


# checking weight
mpg_weight = smf.ols(formula='mpg ~ weight', data=mpg_data).fit()
mpg_weight.summary()


# In[39]:


# Weight plot 
sns.set_style("darkgrid")

sns.lmplot(y='mpg', x='weight', data=mpg_data)


# In[41]:


# horsepower 
mpg_horsepower = smf.ols(formula='mpg ~ horsepower', data=mpg_data).fit()
mpg_horsepower.summary()


# In[42]:


# horsepower plot 
sns.set_style("darkgrid")

sns.lmplot(y='mpg', x='horsepower', data=mpg_data)


# In[45]:


# model_year 
mpg_model_year = smf.ols(formula='mpg ~ model_year', data=mpg_data).fit()
mpg_model_year.summary()


# In[46]:


# model_year plot 
sns.set_style("darkgrid")

sns.lmplot(y='mpg', x='model_year', data=mpg_data)


# In[73]:


# Weight provided the best fit model for simple regression with an R-Squared of 0.691 adjusted and P-Value below 0.05
mpg_weight.params


# In[85]:


# create a DataFrame with the minimum and maximum values of Weight
# these values will be be used in the built model to predict the Price
mpg_data_new = pd.DataFrame({'weight': [mpg_data.weight.min(), mpg_data.weight.max()]})
mpg_data_new.head()


# In[89]:


# Predicted mpg at min and max weight 
preds = mpg_weight.predict(mpg_data_new)
preds


# In[88]:


# Mpg a nd plot the observed data
mpg_data.plot(kind='scatter', x='weight', y='mpg')


# In[90]:


# Now, plot a line over the points that uses just the two points
mpg_data.plot(kind='scatter', x='weight', y='mpg')
# this code overlays a straight line between the the coordinates created by mpg_data_new and preds
plt.plot(mpg_data_new, preds, c='red', linewidth=2)


# In[80]:


# Thus we could expect a car weighing 3000 to have the following fuel consumption in miles per gallon
x = -0.007677
intercept = 46.317364
y = intercept + (3000*x)
y


# In[68]:


# Using Multi-regression to see if we could have a better model
# after testing few combinations, below was the best due to higher adjusted R squared. Additionally, I have avoided having displacement, cylinders, horsepower and weight together due multicollinearity
mpg_model_year_weight = smf.ols(formula='mpg ~ model_year + weight + origin', data=mpg_data).fit()
mpg_model_year_weight.summary()


# In[65]:


# the above table as a heatmap to easily identify relationships
sns.heatmap(mpg_data_corr, 
        xticklabels=mpg_data_corr.columns,
        yticklabels=mpg_data_corr.columns, annot=True)

