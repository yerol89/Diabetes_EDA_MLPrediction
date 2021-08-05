# Diabetes_EDA_MLPrediction

Hello Everyone,\
In this repository, I am supposed to develop a model to predict whether a person is diabetic.\
For this dataset, the variable **Outcome** is a categorical variable. So, this is problem is a **CLASSIFICATION PROBLEM**\
In this ReadMe part, I will try to explain each part of my source code and give some detail about the theory behind each part.

## 1. EXPLORATORY DATA ANALYSIS
![Exploratory Data Analysis](https://media-exp1.licdn.com/dms/image/C4E12AQHlhF7SooEmDA/article-cover_image-shrink_600_2000/0/1602837697395?e=1632960000&v=beta&t=PStG3FAslTNIMW4_W2D4C3kUG5qGHFbs0nfNbIyL8uI)

|  Variable Name |        Definition    |            Key           |
| -------------- | ---------------------| -------------------------|
|   Outcome      |  Diabetic or Healthy |0 = Healthy, 1 = Diabetic |
|   Glucose       | Plasma glucose concentration of 2 hours in an oral glucose tolerance test | - |
|   BMI         |  Body Mass Index |             -            |
|   Insulin          |2-Hour serum insulin (mu U/ml)|       -     |
|   Age          |      Age in years    |             -            |
|   Pregnancies        |# of pregnancies|             -            |
|   SkinThickness        |Skin thickness|             -            |
|   DiabetesPedigreeFunction       |Diabetes Pedigree Function |             -            |
|   BloodPressure         |      Diastolic blood pressure (mm Hg)  |             -            |

At line 29 we have a method named **check_df** and this method returns the shape of the dataframe, types of the variables, sum of missing values for each variable and quantile values.
At line 34 we have **grab_col_names** method. Under **Helpers** folder we have **eda.py** file. Inside this python file you can see the details about this method inside a docstring.

`check_df(df_diabetes)`
- - -
This dataset has 768 observations with 9 variables.

|  Variable Name |   Variable Types | Missing Values |
| -------------- | ---------------- | ---------------|
|   Outcome  |      int64       |       0        |
|   Glucose     |    float64       |       0        |
|   BMI       |      float64       |       0        |
|   Insulin         |      float64      |       0        |
|   Age          |      float64      |       0        |
|   Pregnancies          |      float64     |     0        |
|   SkinThickness        |      float64      |       0        |
|   DiabetesPedigreeFunction        |      float64       |       0        |
|   BloodPressure       |      float64      |       0        |

## 2. DATA PRE_PROCESSING
   
   ### Dealing With Outliers
   If `check_outlier(df_diabetes, col)` method returns for a variable it means that there exist an outlier value in this variable.\
   Then I use `replace_with_thresholds(df_diabetes, col)` method to transform this outliers with threshold values.\
   But how should we determine these thresholds?\
   I use **Interquartile Range Method** for this. You can find an excellent explanation on this link [IQR](https://online.stat.psu.edu/stat200/lesson/3/3.2)
   
   ### Dealing With Missing Values
   `missing_values_table(df_diabetes)` method gives the missing values.\
   I generally fill the missing values with the median or mode values but it depends on the variable and the dataset of course.
    
  ### Feature Engineering
  `
  corr_matrix = df_titanic.corr()
  print(corr_matrix["Outcome"].sort_values(ascending=False))
  `\
  The code snippet given above, returns us the table seen below.\ 
  Using the values in this table we can easily understand that **"Glucose, BMI, Age"** have high correlation with our target variable "Outcome".\
  So I used these variables to create new features.
  Glucose                    0.493
BMI                        0.313
Insulin                    0.266
Age                        0.243
Pregnancies                0.220
SkinThickness              0.220
DiabetesPedigreeFunction   0.185
BloodPressure              0.169
|  Variable Name |   Correlation Value |
| -------------- | --------------------| 
| Glucose    |      0.493       | 
|   BMI     |      0.313       | 
|   Insulin       |      0.266       | 
|   Age     |      0.243       | 
|   Pregnancies      |      0.220       |
|   SkinThickness  |      0.220       | 
|   DiabetesPedigreeFunction  |      0.185       | 
|   BloodPressure  |      0.169       | 

  ### Encoding
  
  Many datasets have non-numerical variables. Although some ML algorithms handle this kind
  of categorical variables, many of them waits for numerical values to fit the model.\
  So one of the most important challenges is turning categorical variables into numerical ones. To achieve this, I used **One Hot Encoder(OHE)** in this repository.\
  **OHE** converts each value into a new variable and assigns 0 or 1 as variable values.\
  In other words, **OHE** map each label to a binary vector
  
## 3. MODEL FITTING
I split my dataset into two parts as test and train sets.\
Then used train set to fit the model.\
I fit several ML algorithms for comparison in this repository.\
**How can we understand or measure our success level of prediction?**\
  ### Performance Measures
  To understand performance measures better we should first talk about the **Confusion Matrix** 
      
  ![Confusion Matrix](https://miro.medium.com/max/1000/1*fxiTNIgOyvAombPJx5KGeA.png)
  True positive and true negatives are the correctly predicted observations and shown in green. 
  We want to reduce the value for false positives and false negatives and they shown in red color.

  **True Positives (TP)** - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes. 
  For example, if actual class value says that a person is diabetic and predicted class tells you the same, this can be considered as a TP value.    
  
  **True Negatives (TN)** - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. 
  For example, if actual class says a person is not diabetic and predicted class tells also the same thing, this can be considered as TN.

  False positives and false negatives, these values occur when your actual class says the opposite of the predicted class.

  **False Positives (FP)** – When actual class is no and predicted class is yes. 
  For example, if actual class says a person is diabetic but predicted class tells you that this person is not diabetic than this is a FP.\ 
  In other words, actually not positive(not diabetic) but predicted as positive(diabetic).
  
  **False Negatives (FN)** – When actual class is yes but predicted class is no. 
  For example, if actual class value says that a person is diabetic and predicted class tells you this person is healthy.\
  In other words, actually positive(diabetic) but predicted as not positive(not diabetic).

  Once you understand these four parameters then we can calculate Accuracy, Precision, Recall and F1 score.

  **Accuracy -** Accuracy is a performance measure and it is a ratio of correctly predicted observation to the total observations. 
  One may think that, if we have high accuracy then our model is best. Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same. Therefore, you have to look at other parameters to evaluate the performance of your model.

  **Accuracy = TP+TN/TP+FP+FN+TN**

  **Precision -** Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 
  The question that this metric answesr is of all people that labeled as diabethic, how many actually diabethic? High precision relates to the low false positive rate.
  In other words, what is the percentage of the people really diabethic, to all the people predicted as diabethic.
  **Precision = TP/TP+FP**

  **Recall (Sensitivity) -** Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. 
  The question recall answers is: Of all the passengers that truly diabethic, how many did we label?
  In other words, what is our model's success level of labeling truly diabethic people out of all really diabethic ones ? 
  **Recall = TP/TP+FN**

  **F1 score -** F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.

  **F1 Score = 2*(Recall * Precision) / (Recall + Precision)**
  
  Even after all these metrics we may need some other methods to evaluate our model success. Especially for the classification problems like our dataset, **ROC/AUC** is a popular metric also.
  I used this metric to measure the performance of my model. If you interested in these metrics and their comparisons, I strongly advice you to read this article on the [link](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc).
  
  
  
  
