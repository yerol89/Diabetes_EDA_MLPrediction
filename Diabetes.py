# Importing libraries and modules
from Helpers.data_prep import *
from Helpers.eda import *
from Helpers.ML_Helpers import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

# Settings to display data on the console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

# I load the Diabetes dataset using "load_diabetes()" method
df_diabetes = load_diabetes()

# "check_df" method returns the shape of dataset, types of variables(categorical, numeric)
# number of missing values and quantiles
check_df(df_diabetes)
df_diabetes.describe().T

# "grab_col_names" method returns us the type of variables
# I will explain this method at "Read Me" part in detail
cat_cols, num_cols, cat_but_car = grab_col_names(df_diabetes)

# "cat_summary" method returns the ratio of each observation for each categorical variable
for col in cat_cols:
    cat_summary(df_diabetes, col)

# CHECKING FOR MISSING VALUES

# "missing_values_table" column returns the variables with missing values, number of them and their ratio
missing_values_table(df_diabetes) # No Missing Value
# But there are unordinary amount of 0 as variable values.
# So we can understand that NA values filled with 0 value.
# We can again turn 0's to NA and then change NA's with median values.
df_diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
    df_diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df_diabetes = df_diabetes.apply(lambda x: x.fillna(x.median()), axis=0)
# Re-control missing values
missing_values_table(df_diabetes) # No Missing Value

# CHECKING FOR OUTLIERS

# We should choose between "transforming the value" or "deleting the value"
# For this dataset, I decided to "transform the outlier values" due to the insufficient number of observations
# I used "check_outliers" method to determine whether numerical variables have outliers or not
for col in num_cols:
    print(check_outlier(df_diabetes, col)) # Returns True-False for each variable => All variables has outliers


# I transformed the outliers with the threshold values.
# I determined this thresholds using "Interquartile Range Method" that I will explain at "Read Me" in detail
# "replace_with_thresholds" method contains the logic of "Interquartile Range Method" also
for col in num_cols:
    replace_with_thresholds(df_diabetes, col) # Changed outliers with threshold values
df_diabetes.head(30)
#####################
# FEATURE ENGINEERING
#####################
# We can see the output of correlation method and level of correlation.
# So I decided to create new features mostly using these variables
corr_matrix = df_diabetes.corr()
print(corr_matrix["Outcome"].sort_values(ascending=False))
'''
Glucose                    0.493
BMI                        0.313
Insulin                    0.266
Age                        0.243
Pregnancies                0.220
SkinThickness              0.220
DiabetesPedigreeFunction   0.185
BloodPressure              0.169
'''


def diabetes_feature_engineering(df):
    df.loc[(df['Glucose'] <= 120) & (df['Age'] <= 30), 'New_Glucose_Age'] = "0"
    df.loc[(df['Glucose'] > 120), 'New_Glucose_Age'] = "1"

    df.loc[(df['Glucose'] >= 140) & (df['BMI'] >= 25), 'New_Glucose_BMI'] = "1"
    df.loc[(df['Glucose'] < 140), 'New_Glucose_BMI'] = "0"

    df.loc[(df['Glucose'] <= 105) & (df['BloodPressure'] <= 80), 'New_Glucose_BloodPressure'] = "0"
    df.loc[(df['Glucose'] > 105), 'New_Glucose_BloodPressure'] = "1"


diabetes_feature_engineering(df_diabetes)

# After creating new features I re-implement all the EDA operations again
cat_cols, num_cols, cat_but_car = grab_col_names(df_diabetes)
for col in cat_cols:
    cat_summary(df_diabetes, col)

#############################################
#One_Hot_Encoding
#############################################
# Datasets consists of both "categorical" and "numerical" values most of the time
# But ML algorithms waits for numerical values to process them
# One Hot Encoding is a way to deal with this problem
# On the next line there is a list comprehension to filter columns for OHE
a = ['Outcome']
ohe_cols = [col for col in df_diabetes.columns if (10 >= df_diabetes[col].nunique() >= 2) & (col not in a)]
df_diabetes = one_hot_encoder(df_diabetes, ohe_cols)
# After OHE, we have many new variables, so we need to split them as categorical and numerical again
cat_cols, num_cols, cat_but_car = grab_col_names(df_diabetes)
df_diabetes.head()


# Feature Scaling With Robust Scaler
scaler = RobustScaler()
df_diabetes[num_cols] = scaler.fit_transform(df_diabetes[num_cols])

##########################
# Fitting Model
##########################

##########################
# Model
##########################
df_diabetes
# Dependent Variable
y = df_diabetes["Outcome"]
# Independent Variables
X = df_diabetes.drop(["Outcome"], axis=1)

# I split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

# Here are some ML algorithms that I will use to predict my target variable
log_model = LogisticRegression(solver='liblinear').fit(X_train, y_train)
rfm = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1, random_state=101,
                            max_features=None, min_samples_leaf=30).fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth=10, random_state=101,
                            max_features=None, min_samples_leaf=15).fit(X_train, y_train)
nb = GaussianNB().fit(X_train, y_train)
lgbm_model = LGBMClassifier(random_state=46)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

lgbm_final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

model_list = [log_model, rfm, dt, nb, lgbm_final_model]

# Confusion Matrix For Each Model
y_pred_confusion_log = log_model.predict(X_test)
y_pred_confusion_rfm = rfm.predict(X_test)
y_pred_confusion_dt = dt.predict(X_test)
y_pred_confusion_nb = nb.predict(X_test)
y_pred_confusion_lgbm = lgbm_final_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred_confusion_log)
plot_confusion_matrix(y_test, y_pred_confusion_rfm)
plot_confusion_matrix(y_test, y_pred_confusion_dt)
plot_confusion_matrix(y_test, y_pred_confusion_nb)
plot_confusion_matrix(y_test, y_pred_confusion_lgbm)

# I wrote a method to display "Accuracy, Precision, Recall and F1 Scores" which I will explain in detail at "Read Me"
# Also I used ROC_AUC Score for prediction


def model_processor():
    for model in model_list:
        y_pred = model.predict(X_train)

        ##########################
        # Evaluation of Predictions
        ##########################

        # Train Accuracy
        print(f"Train Accuracy for {model}: {accuracy_score(y_train, y_pred)}")

        # Test Accuracy

        # For AUC Score  y_prob
        y_prob = model.predict_proba(X_test)[:, 1]

        # For the other metrics y_pred
        y_pred = model.predict(X_test)

        # ACCURACY
        print(f"Test Accuracy for {model}: {accuracy_score(y_test, y_pred)}")

        # PRECISION
        print(f"Precision Score for {model}: {precision_score(y_test, y_pred)}")

        # RECALL
        print(f"Recall Score for {model}: {recall_score(y_test, y_pred)}")

        # F1
        print(f"F1 Score for {model}: {f1_score(y_test, y_pred)}")

        # ROC CURVE
        plot_ROC_curve(model, X_test, y_test)

        # AUC
        print(f"AUC Score for {model}: {roc_auc_score(y_test, y_prob)}")

        # Classification report
        print(f"CLASSIFICATION REPORT FOR {model}")
        print(classification_report(y_test, y_pred))

    def plot_importance(model, features, num=len(X), save=False):
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
        plt.figure(figsize=(13, 10))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[0:num])
        plt.title('Features')
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig('importances.png')

    for models in [rfm, dt, lgbm_final_model]:
        print(f"IMPORTANCE PLOT FOR {models}")
        plot_importance(models, X, 15)


model_processor()

