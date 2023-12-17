#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns",None)
pd.set_option("display.max_colwidth",200)
pd.set_option("display.max_rows", None)


# In[3]:


df = pd.read_csv('Leads.csv')
df.head()


# In[4]:


df.shape


# ## AUTO-EDA

# In[5]:


get_ipython().system('pip install sweetviz')


# In[6]:


import sweetviz as sv
sweet_report = sv.analyze(df)
sweet_report.show_html('sweet_report.html')


# In[7]:


sns.pairplot(df)
plt.show()


# In[ ]:





# In[8]:


df.columns


# # DATA CLEANING AND PREPARATION

# In[9]:


df.isnull().sum().sort_values(ascending=False)


# it shows that there are a lot of columns which have high number of missing values. These columns are not useful. Since, there are 9240 datapoints in our dataframe, We should eliminate the columns having >=3000 missing values as they are of not significant to it.

# LET"S REMOVE THE COLUMN WHIC HAVE >=3000 MISSING VALUES.

# In[10]:


for c in df.columns:
    if df[c].isnull().sum()>=3000:
        df.drop(c,axis=1,inplace=True)


# In[11]:


df.isnull().sum().sort_values(ascending=False)


# CHECKING THE VALUE COUNTS OF CITY COLOM

# In[12]:


df['City'].value_counts(dropna=False).sort_values(ascending=False)


# AS IT MIGHT NOT BE USEFUL COLOM 'CITY' SO WE HAVE TO DROP IT FROM OUR DATASET.

# In[13]:


df.drop(['City'], axis=1, inplace=True)


# CHECKING THE COUNTS FOR THE COUNTRY COLOM

# In[14]:


df['Country'].value_counts(dropna=False)


# In[15]:


df.drop(['Country'],axis=1,inplace=True)


# In[16]:


df.isnull().sum().sort_values(ascending=False)


# In[17]:


df.drop(['Lead Profile', 'How did you hear about X Education'], axis = 1, inplace = True)


# In[18]:


df.isnull().sum().sort_values(ascending=False)


# In[19]:


x_edu = df[['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']]

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
transformedx_edu = pd.DataFrame(pt.fit_transform(x_edu))
transformedx_edu.columns = x_edu.columns
transformedx_edu.head()


# In[20]:


df.drop(['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
            'Update me on Supply Chain Content', 'Get updates on DM Content', 
            'I agree to pay the amount through cheque'], axis = 1, inplace = True)


# In[21]:


df['What matters most to you in choosing a course'].value_counts()


# In[22]:


df.drop(['What matters most to you in choosing a course'], axis = 1, inplace=True)


# In[23]:


df.isnull().sum().sort_values(ascending=False)


# LET'S ONLY DROP THE MISSING VALUES OF 'WHAT IS YOUR OCCUPATION' COLOM

# In[24]:


# Dropping the null values rows in the column 'What is your current occupation'
df = df[~pd.isnull(df['What is your current occupation'])]


# In[25]:


sns.heatmap(df.corr(), annot=True,cmap="RdYlGn", robust=True,linewidth=0.1, vmin=-1 )
plt.show()


# In[26]:


df.isnull().sum().sort_values(ascending=False)


# In[27]:


df = df[-pd.isnull(df['TotalVisits'])]
df = df[-pd.isnull(df['Page Views Per Visit'])]
df = df[-pd.isnull(df['Lead Source'])]
df = df[-pd.isnull(df['Last Activity'])]
df = df[-pd.isnull(df['Specialization'])]


# In[28]:


df.isnull().sum().sort_index(ascending=False)


# NOW OUR DATA IS CLEAN AND NO NULL VALUES ARE THERE

# In[29]:


len(df.index)


# In[30]:


100*len(df.index)/9240


# NOW WE HAVE 69% OF ROWS WHICH ARE CLEAN FOR PROCESSING

# In[31]:


df.head()


# In[32]:


df.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[33]:


df.head()


# In[34]:


df.shape


# In[35]:


df.info()


# In[36]:


df.describe()


# ## DUMMY VARIABLE CREATION

# In[37]:


temp = df.loc[:, df.dtypes == 'object']
temp.columns


# In[38]:


dummy = pd.get_dummies(df[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                              'What is your current occupation','A free copy of Mastering The Interview', 
                              'Last Notable Activity']], drop_first=True)


# In[39]:


df = pd.concat([df,dummy], axis=1)
df.head()


# In[40]:


# Creating dummy variable separately for the variable 'Specialization' since it has the level 'Select' 
# which is useless so we
# drop that level by specifying it explicitly

dummy_spl = pd.get_dummies(df['Specialization'], prefix = 'Specialization')
dummy_spl = dummy_spl.drop(['Specialization_Select'], 1)
df = pd.concat([df, dummy_spl], axis = 1)


# In[41]:


df.shape


# In[42]:


# Dropping the variables for which the dummy variables have been created
df = df.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                   'Specialization', 'What is your current occupation',
                   'A free copy of Mastering The Interview', 'Last Notable Activity'], 1)


# In[43]:


df.head()


# # TRAIN-TEST SPLIT

# In[44]:


import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics

from sklearn.metrics import classification_report,recall_score,roc_auc_score,roc_curve,accuracy_score,precision_score,precision_recall_curve,confusion_matrix

from sklearn.preprocessing import LabelEncoder


# In[45]:


X = df.drop(['Converted'],1)
X.head()


# In[46]:


y = df['Converted']
y.head()


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[48]:


print(X.shape)
print(y.shape)


# # SCALING

# In[49]:


scaler = MinMaxScaler()

X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

X_train.head()


# # STEP 2 : MODEL BUILDING

# In[50]:


logreg = LogisticRegression()


# In[51]:


rfe = RFE(estimator=logreg, n_features_to_select=15) # running RFE with 15 vars
rfe


# In[52]:


rfe = rfe.fit(X_train,y_train)
rfe


# In[53]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[54]:


# Putting all the columns selected by RFE in the variable 'col'


col = X_train.columns[rfe.support_]
col


# In[55]:


X_train = X_train[col]
X_train


# Now we have all the variables selected by RFE and since we care about the statistics part, i.e. the p-values and the VIFs, we have to use these variables to create a logistic regression model using statsmodels.

# ## MODEL-1

# In[56]:


# import stats library
from scipy import stats
import statsmodels.api as sm


# In[57]:


X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# ## CHECKING VIF

# In[58]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ## MODEL-2

# In[59]:


X_train.drop('Lead Origin_Lead Add Form', axis = 1, inplace = True)


# In[60]:


logm2 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm2.fit().summary()


# ## CHECKING VIF

# In[61]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ## MODEL-3

# In[62]:


X_train.drop('Last Notable Activity_Had a Phone Conversation', axis = 1, inplace = True)


# In[63]:


logm3 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm3.fit().summary()


# ## CHECKING VIF

# In[64]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ## MODEL-4

# In[65]:


X_train.drop('What is your current occupation_Housewife', axis = 1, inplace = True)


# In[66]:


logm4 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm4.fit().summary()


# ## CHECKING VIF

# In[67]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # STEP 3 : MODEL EVALUTION

# In[68]:


# Use 'predict' to predict the probabilities on the train set

y_train_pred = res.predict(sm.add_constant(X_train_sm))
y_train_pred[:10]


# In[69]:


# Reshaping it into an array

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# Creating a new dataframe containing the actual conversion flag and the probabilities predicted by the model

# In[70]:


y_train_pred_final = pd.DataFrame({'Converted(ACTUAL)':y_train.values, 'Conversion_Probability(PREDICTED)':y_train_pred})
y_train_pred_final.head()


#  Creating new column 'Predicted' with 1 if Paid_Prob > 0.5 else 0

# In[71]:


y_train_pred_final['Predicted'] = y_train_pred_final['Conversion_Probability(PREDICTED)'].map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# ### NOW WE CAN CREATE CONFUSION MATRIX

# In[72]:


confusion = metrics.confusion_matrix(y_train_pred_final['Converted(ACTUAL)'], y_train_pred_final['Predicted'])
print(confusion)


# In[73]:


metrics.accuracy_score(y_train_pred_final['Converted(ACTUAL)'], y_train_pred_final['Predicted'])


# In[74]:


#OTHER PARAMS

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[75]:


print(TP,TN,FP,FN)


# In[76]:


print('ACCURACY')
TP/(TP+FN)


# In[77]:


print('SPECIFICITY')
TN/(TN + FP)


# ### Finding the Optimal Cutoff
# Now 0.5 was just arbitrary to loosely check the model performace. But in order to get good results, you need to optimise the threshold. So first let's plot an ROC curve to see what AUC we get.

# In[78]:


# ROC function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[79]:


fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final['Converted(ACTUAL)'],
                    y_train_pred_final['Conversion_Probability(PREDICTED)'], 
                                         drop_intermediate=False)


# In[80]:


# Calling the ROC function

draw_roc(y_train_pred_final['Converted(ACTUAL)'], y_train_pred_final['Conversion_Probability(PREDICTED)'])


# The area under the curve of the ROC is 0.86 which is quite good. So we seem to have a good model. Let's also check the sensitivity and specificity tradeoff to find the optimal cutoff point.

# In[81]:


# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final['Conversion_Probability(PREDICTED)'].map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[82]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['PROBABILITY','ACCURACY','SENSITIVITY','SPECIFICITY'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final['Converted(ACTUAL)'], y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[83]:


plt.figure(figsize=(5,4))
cutoff_df.plot.line(x='PROBABILITY', y=['ACCURACY','SENSITIVITY','SPECIFICITY'])
plt.show()


# As you can see that around `0.42`, you get the optimal values of the three metrics. So let's choose 0.42 as our cutoff now.

# # STEP 4 : MAKING PREDICTIONS ON TEST SET

# In[84]:


# Scaling the test set as well using just 'transform'

X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] =  scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[85]:


# Selecting the columns in X_train for X_test as well

X_test = X_test[col]
X_test.head()


# In[87]:


# Adding a constant to X_test

X_test_sm = sm.add_constant(X_test[col])
X_test_sm


# In[88]:


# Dropping the required columns from X_test as well

X_test.drop(['Lead Source_Reference', 'What is your current occupation_Housewife', 
             'What is your current occupation_Working Professional', 
                     'Last Notable Activity_Had a Phone Conversation'], 1, 
                                inplace = True)


# In[91]:


# Make predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(sm.add_constant(X_test_sm))
y_test_pred[:10]


# In[94]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1[:10]


# In[97]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
y_test_df[:10]


# In[98]:


# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[101]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[103]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[105]:


# Make predictions on the test set using 0.45 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)
y_pred_final.head()


# NOW WE CAN CALCULATE SOME CRITERIA

# In[107]:


# Let's check the overall accuracy
print('ACCURACY')
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[109]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
print(confusion2)


# In[110]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[112]:


print(TP,TN,FP,FN)


# In[113]:


print('SENSITIVITY')

TP / (TP + FN)


# In[114]:


print('SPECIFICITY')

TN / (TN +FP)


# ### PRECESION-RECALL

# In[117]:


confusion = metrics.confusion_matrix(y_train_pred_final['Converted(ACTUAL)'], y_train_pred_final.Predicted )
confusion


# In[118]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# ### PRECESION

# In[119]:


print('PRECESION')

TP / (TP + FP)


# ### RECALL

# In[120]:


print('RECALL')

TP / (TP + FN)


# ### PRECESION AND RECALL TRADEOFF

# In[122]:


y_train_pred_final['Converted(ACTUAL)'], y_train_pred_final.Predicted


# In[125]:


p, r, thresholds = precision_recall_curve(y_train_pred_final['Converted(ACTUAL)'], y_train_pred_final['Conversion_Probability(PREDICTED)'])


# In[126]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# # STEP 5 : MAKING PREDICTIONS

# In[130]:


y_test_pred = res.predict(sm.add_constant(X_test_sm))
y_test_pred[:10]


# In[132]:


y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1[:10]


# In[134]:


y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[135]:


# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[137]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[138]:


# Making predictions on the test set using 0.44 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.44 else 0)


# In[139]:


y_pred_final.head()


# ## CHECKING CRITERIA

# In[140]:


# Let's checking the overall accuracy

metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[141]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[142]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[144]:


print(TP, TN, FP, FN)


# In[145]:


print('PRECESION')

TP / float(TP + FP)


# In[146]:


print('RECALL')

TP /float(TP + FN)


# In[ ]:




