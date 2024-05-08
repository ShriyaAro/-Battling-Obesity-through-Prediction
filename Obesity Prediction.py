#!/usr/bin/env python
# coding: utf-8

# # 
# This code performs a comprehensive data preprocessing and machine learning pipeline for a classification problem, using various datasets and machine learning models. Here's a step-by-step summary:
# 
# Imports necessary libraries for data manipulation, visualization, model training, and evaluation.
# 
# Suppresses warnings to avoid cluttering the output with unnecessary warning messages.
# 
# Reads datasets including training data, test data, an original dataset, and a sample submission file.
# 
# #Data Preprocessing:
# 
# Drops the 'id' column from the training data.
# Concatenates the training data with the original dataset to enhance the training set and drops duplicates.
# Adds a 'BMI' column to the training, test, and original datasets, calculated from weight and height.
# 
# #Exploratory Data Analysis (EDA):
# 
# Visualizes the distribution of BMI across the training dataset.
# Generates a correlation matrix heatmap to understand the relationships between variables.
# Further Visualization:
# 
# Creates functions to visualize the distribution of target and categorical variables across datasets.
# Visualizes the distribution of the 'NObeyesdad' variable and other categorical variables.
# 
# #Feature Engineering:
# 
# Scales numerical variables using StandardScaler to normalize data.
# Encodes categorical variables using LabelEncoder to convert them into numerical values.
# 
# #Model Preparation:
# 
# Splits the features (X) from the target (y) in the training dataset.
# Further splits the training data into training and validation sets.
# 
# #Model Training with LightGBM:
# 
# Defines parameters for a LightGBM classifier and trains the model on the training set.
# 
# #Hyperparameter Optimization:
# 
# Uses optuna for optimizing the thresholds applied to the model's probability predictions to maximize accuracy.
# 
# #Prediction and Submission File Creation:
# 
# Applies the optimized thresholds to the model's predictions on the test dataset.
# Inverses the label encoding to get the original labels for the predictions.
# Creates a submission file with predictions for uploading to a competition or for evaluation purposes.
# 
# The code is comprehensive, covering multiple steps from data loading and preprocessing, exploratory data analysis, feature engineering, model training with parameter optimization using LightGBM, and finally, preparing a submission file. It showcases a thorough approach to tackling a machine learning classification problem, emphasizing the use of gradient boosting models and the importance of handling categorical and numerical data appropriately.

# # Import Libraries

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.getLogger('lightgbm').setLevel(logging.INFO)
logging.getLogger('lightgbm').setLevel(logging.ERROR)


# # Data Loading

# In[32]:


##data loading
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
original_data = pd.read_csv("ObesityDataSet.csv")
sample_submission_data = pd.read_csv("sample_submission.csv")


# In[33]:


#drop id
train_data = train_data.drop("id", axis=1)
train_data = pd.concat([train_data, original_data], ignore_index=True)
train_data = train_data.drop_duplicates()
train_data.shape


# # Featuring BMI for better model

# In[34]:


# Calculate BMI and add to datasets
train_data['BMI'] = train_data['Weight'] / (train_data['Height']**2)
test_data['BMI'] = test_data['Weight'] / (test_data['Height']**2)
original_data['BMI'] = original_data['Weight'] / (original_data['Height']**2)


# In[62]:


train_data.head()


# # Exploring Different Visualization to Understand Variable Relationships 

# In[63]:


# EDA
# Visualize the distribution of BMI
plt.figure(figsize=(10, 6))
sns.histplot(train_data['BMI'], bins=30, kde=True)
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.axvline(x=train_data['BMI'].mean(), color='red', linestyle='--', label='Mean BMI')
plt.axvline(x=train_data['BMI'].median(), color='green', linestyle='--', label='Median BMI')
plt.legend()
plt.show()


# Correlation matrix
corr = train_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[64]:


##BMI distribution plot
plt.figure(figsize=(14, 7))
sns.histplot(data=train_data, x='BMI', hue='NObeyesdad', element='step', kde=True)
plt.title('BMI Distribution with Obesity Level')
plt.xlabel('BMI')
plt.ylabel('Density')


# In[57]:


# Plotting box plots for Age, Height, and Weight
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
sns.boxplot(y=train_data['Age'])
plt.title('Box Plot of Age')

plt.subplot(1, 3, 2)
sns.boxplot(y=train_data['Height'])
plt.title('Box Plot of Height')

plt.subplot(1, 3, 3)
sns.boxplot(y=train_data['Weight'])
plt.title('Box Plot of Weight')

plt.show()


# In[59]:


from scipy import stats
# Calculating Z-scores of Age, Height, and Weight
z_scores = np.abs(stats.zscore(train_data[['Age', 'Height', 'Weight']]))
threshold = 3
outliers_z = np.where(z_scores > threshold)

print("Outlier indices based on Z-score:")
print(np.unique(outliers_z[0]))  # Printing unique indices of outliers


# In[61]:


# Calculating IQR for Age, Height, and Weight
Q1 = train_data[['Age', 'Height', 'Weight']].quantile(0.25)
Q3 = train_data[['Age', 'Height', 'Weight']].quantile(0.75)
IQR = Q3 - Q1

# Determining outliers using IQR rule
outliers_iqr = ((train_data[['Age', 'Height', 'Weight']] < (Q1 - 1.5 * IQR)) | (train_data[['Age', 'Height', 'Weight']] > (Q3 + 1.5 * IQR))).any(axis=1)

print("Number of outliers detected using IQR:", outliers_iqr.sum())
print("Outlier indices based on IQR:", train_data[outliers_iqr].index.tolist())


# # Setting the plot distribution for detailed EDA for multiple variables

# In[36]:


##Setting the plot distribution for detailed EDA for multiple variables

import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(dataframe, target_column, dataset_type='train'):
    # Calculate value counts
    value_counts = dataframe[target_column].value_counts()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot on the first subplot using 'cividis' palette
    sns.barplot(x=value_counts.index, y=value_counts.values, palette="cividis", ax=ax1)
    ax1.set_xlabel(target_column, fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    # Add data labels above each bar
    for index, value in enumerate(value_counts):
        ax1.text(index, value, str(value), ha='center', va='bottom', fontsize=10)

    # Pie plot on the second subplot with 'cividis' colors
    ax2.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', colors=sns.color_palette("cividis", len(value_counts)))
    ax2.axis('equal')

    # Main title for the figure based on dataset type
    fig.suptitle(f'Comparison of {target_column} Distribution in {dataset_type.capitalize()} Dataset', fontsize=18)
        
    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()


# In[37]:


#Comparison of obesity distribution in Train dataset
plot_distribution(train_data, 'NObeyesdad')


# # 
# The plots depicts two graphs comparing the distribution of various weight categories in a dataset, with Obesity Type III being the most common category in the bar chart and Obesity Type I having the highest percentage in the pie chart.

# In[38]:


#Comparison of obesity distribution in Original dataset
plot_distribution(original_data, 'NObeyesdad')


# # 
# The plotpresents two visualizations of a original dataset's distribution across various weight categories, showing a relatively balanced distribution with Obesity Type I as the most prevalent category in both the bar and pie charts.

# In[39]:


##Defining Categorical & Numerical variables and specifying 
cat_vars = train_data.select_dtypes(exclude=np.number).columns.tolist()
num_vars = train_data.select_dtypes(include=np.number).columns.tolist()

cat_vars.remove('NObeyesdad')

print("Continuous Variables:", num_vars)
print("Categorical Variables:", cat_vars)


# In[40]:


#Comparison of different variable distribution like gender, SMOKE, etc. distribution in Train dataset
for column in cat_vars:
    plot_distribution(train_data, column)


# # Summary
# 
# Gender Distribution: This shows an almost equal distribution between female and male participants in the dataset, with females slightly outnumbering males.
# 
# Family History with Overweight: A large majority of the dataset, 82%, have a family history of overweight, which may suggest a genetic or lifestyle pattern within families related to weight issues.
# 
# FAVC (Frequent Consumption of High Caloric Food): A significant majority, 91.2%, of the participants frequently consume high-caloric food, which might be a factor influencing weight categories within the dataset.
# 
# CAEC (Consumption of Food Between Meals): The majority, 84.4%, sometimes consume food between meals, with a very small percentage always or frequently doing so.
# 
# SMOKE: Only a small fraction, 1.3%, of the dataset are smokers, suggesting that smoking habits are not common among the participants.
# 
# SCC (Calories Consumption Monitoring): A vast majority, 96.6%, do not monitor their calorie consumption, which may have implications for weight management and overall health.
# 
# CALC (Alcohol Consumption): A large portion of the dataset, 72%, sometimes consume alcohol, with a very small number doing so frequently or always.
# 
# MTRANS (Transportation Used): Public transportation is the most commonly used mode of transport, according to 79.9% of the dataset, with automobiles, walking, motorbikes, and bikes being less common.

# In[55]:


##Histogram of obesity with other variables like age, height, weight etc. 

def plot_histograms_and_density(dataframe, columns):
    for column in columns:
        fig, ax = plt.subplots(figsize=(20, 10))
        fig = sns.histplot(data=train_data, x=column, hue="NObeyesdad", bins=50, kde=True)
        plt.ylim(0,1000)
        plt.show()
        
plot_histograms_and_density(train_data, num_vars)


# # Histograms overlaid with kernel density estimates (KDE) depicting the distribution of various physical and lifestyle-related attributes across different weight categories. 
# Age Distribution: This histogram shows the age distribution for different weight categories. Peaks and distributions vary, indicating that certain ages are more associated with specific weight statuses.
# 
# Height Distribution: Similar to the age distribution, this histogram shows the height distribution across different weight categories. The range of heights varies among the groups, with some categories like 'Obesity Type I' having a wider spread than others.
# 
# Weight Distribution: This graph illustrates the actual weight distribution among the categories. It is noticeable that weight increases as one moves from 'Insufficient Weight' to 'Obesity Type III'.
# 
# FCVC (Frequency of Consumption of Vegetables): This variable likely indicates how often individuals consume vegetables. The histogram and KDE show the count of individuals across the weight categories and how frequently they eat vegetables.
# 
# NCP (Number of Main Meals): This graph likely shows the distribution of the number of main meals individuals eat in a day, distributed across the weight categories.
# 
# CH2O (Water Drinking): This histogram probably represents the amount of water consumption, with varying distributions across the weight categories.
# 
# FAF (Physical Activity Frequency): This histogram displays the frequency of physical activity across different weight categories, which is a critical factor in weight management.
# 
# TUE (Time Using Technology Devices): This variable likely refers to the time spent using technology devices. The distribution across weight categories might suggest a relationship between sedentary behavior and weight.
# 
# BMI (Body Mass Index): This is a well-known measure that calculates body fat based on height and weight. The histogram shows the BMI distribution for different weight categories, with each category having a distinct range of BMI values.

# In[42]:


num_cols = list(train_data.select_dtypes(exclude=['object']).columns)
cat_cols = list(train_data.select_dtypes(include=['object']).columns)

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns)
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)

num_cols_test = [col for col in num_cols_test if col not in ['id']]


# # Feature Engineering

# In[43]:


##Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols_test] = scaler.transform(test_data[num_cols_test])


# In[44]:


# encoding object datatype
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
object_columns = train_data.select_dtypes(include='object').columns.difference(['NObeyesdad'])

for col_name in object_columns:
    if train_data[col_name].dtypes=='object':
        train_data[col_name]=labelencoder.fit_transform(train_data[col_name])
        
for col_name in test_data.columns:
    if test_data[col_name].dtypes=='object':
        test_data[col_name]=labelencoder.fit_transform(test_data[col_name])


# In[45]:


X = train_data.drop(['NObeyesdad'], axis=1)
y = train_data['NObeyesdad']
y = labelencoder.fit_transform(y)
X_test = test_data.drop(["id"],axis=1)


# In[46]:


#data partition
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)


# # Set parameters for a LightGBM multi-class classification model, initialize the model with these parameters, trains it on the provided training data, and then makes class predictions and probability predictions on the validation set.

# In[47]:


param = {"objective": "multiclass",          
    "metric": "multi_logloss",          
    "verbosity": -1,                    
    "boosting_type": "gbdt",            
    "random_state": 42,       
    "num_class": 7,                     
    'learning_rate': 0.030962211546832760,  
    'n_estimators': 500,                
    'lambda_l1': 0.009667446568254372,  
    'lambda_l2': 0.04018641437301800,   
    'max_depth': 10,                    
    'colsample_bytree': 0.40977129346872643,  
    'subsample': 0.9535797422450176,   
    'min_child_samples': 26}

model_lgb = lgb.LGBMClassifier(**param,verbose=100)
model_lgb.fit(X_train, y_train)
pred_lgb = model_lgb.predict(X_val)
pred_proba = model_lgb.predict_proba(X_val)


# # Optuna optimization process to find the best thresholds for converting predicted probabilities to class labels in order to maximize the accuracy of a classification model.

# In[48]:


import optuna

def objective(trial):
    # Define the thresholds for each class
    thresholds = {}
    for i in range(num_classes):
        thresholds[f'threshold_{i}'] = trial.suggest_uniform(f'threshold_{i}', 0.0, 1.0)

    # Apply the thresholds to convert probabilities to predictions
    y_pred = apply_thresholds(pred_proba, thresholds)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy  

def apply_thresholds(y_proba, thresholds):
    # Apply the specified thresholds to convert probabilities to predicted labels
    y_pred_labels = np.argmax(y_proba, axis=1)
    for i in range(y_proba.shape[1]):
        y_pred_labels[y_proba[:, i] > thresholds[f'threshold_{i}']] = i

    return y_pred_labels


# In[49]:


num_classes = 7
pred_proba = pred_proba  # Example: replace with actual y_pred_proba
y_val = y_val  # Example: replace with actual y_val

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# # Get the best thresholds
best_thresholds = study.best_params
print("Best Thresholds:", best_thresholds)


# In[50]:


threshold= {'threshold_0': 0.724201213234911, 'threshold_1': 0.6161299800571379, 'threshold_2': 0.29138887902587174, 'threshold_3': 0.3145837593497076, 'threshold_4': 0.8469398340837189, 'threshold_5': 0.6800824438387787, 'threshold_6': 0.35886959729223455}


# In[51]:


test_label = model_lgb.predict_proba(X_test)
test_label = apply_thresholds(test_label, threshold)


# In[52]:


#Final prediction and csv file creation
pred = labelencoder.inverse_transform(test_label)
submission = pd.DataFrame({'id': test_data.id, 'NObeyesdad': pred})
submission.to_csv('Comp1_sz9461.csv', index=False)


# # The End

# # Note:
# Kaggle Username: The username "Bunny_Panda" is used for participating in Kaggle competitions. This username was chosen to maintain privacy on the Kaggle platform, and any associated rankings or scores are tied to this pseudonym.
# 
# Name Change Disclosure: It's important to note that "Bunny_Panda" is not the participant's real name. This alias was created to address privacy concerns on Kaggle, ensuring personal information remains confidential.
# 
# Leaderboard Scores: For clarification, the performance metrics associated with "Bunny_Panda" on Kaggle reflect a Private Score of 631 and a Public Score of 173. These scores indicate the model's performance as ranked among other competitors in the Kaggle leaderboard.
# 
# Learning Experience: The Kaggle competition provided a rich learning environment. It offered the chance to delve into multiple Machine Learning methodologies and to hone skills in data visualization, which are essential for insightful data analysis.
# 
# Future Competitions: There is an eagerness to engage in upcoming Kaggle competitions. The experience gained is invaluable, and there is anticipation for the growth and challenges that future contests will bring.
# 
# Model Performance: A draft code version achieved a high accuracy of 96% on the validation dataset, which was a significant accomplishment. However, this same code, when tested against the Kaggle test dataset, resulted in a lower accuracy of 90.859%.
# 
# Final Submission Accuracy: In comparison, the final submitted version of the model showed an improved accuracy, achieving a score of 92.196%. This emphasizes the iterative process of model tuning and the importance of testing the model against various datasets to ensure its generalizability and robustness.
# 
# Accuracy Discrepancy: The difference in accuracy between the validation dataset and the Kaggle test dataset underscores the complexity of Machine Learning models. It highlights the necessity of thorough testing on different datasets to validate model performance and avoid overfitting to the validation data.
# 
# Draft Code Insights: The draft code that initially showed promising results on the validation data but underperformed on the test dataset serves as a reminder that a model's performance can vary depending on the dataset. It's a testament to the iterative nature of Machine Learning, where continuous improvement and adjustments are often required to achieve optimal results.

# # Will also be submitting the draft code for further feedback

# In[ ]:




