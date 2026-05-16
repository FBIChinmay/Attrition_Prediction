import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv('/content/HR-Employee-Attrition.csv')
data
# check number of rows and columns
data.shape
# print first 5 rows
data.head()
# print last 5 rows
data.tail()
# check info
data.info()
# check data types
data.dtypes
# print numerical columns
num_data = data.select_dtypes(include=['int64','float64'])
num_data
# print categorical columns
cat_data = data.select_dtypes(include=['object'])
cat_data
# perform statistical analysis
data.describe()
# find unique values
for x in cat_data:
  print(x ,'-->', data[x].unique())
# find value_counts
for x in cat_data:
  print(data[x].value_counts())
  print()
data.isnull().sum()
data.duplicated().sum()
data["Attrition"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    figsize=(5,5)
)

plt.title("Attrition Distribution")
plt.ylabel("")
plt.show()
sns.countplot(x="Department", data=data)
plt.title("Department Count")
plt.xticks(rotation=45)
plt.show()
sns.histplot(data["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()
sns.boxplot(x=data["MonthlyIncome"])
plt.title("Monthly Income Boxplot")
plt.show()
sns.barplot(x="Department", y="MonthlyIncome", data=data)
plt.title("Average Income by Department")
plt.xticks(rotation=45)
plt.show()
sns.pairplot(
    data[["Age","MonthlyIncome","TotalWorkingYears","YearsAtCompany","Attrition"]],
    hue="Attrition"
)

plt.show()
plt.figure(figsize=(20,10))

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()
pd.crosstab(data["Gender"], data["Attrition"]).plot(
    kind="pie",
    subplots=True,
    autopct="%1.1f%%",
    figsize=(10,5)
)

plt.title("Attrition by Gender")
plt.show()
sns.countplot(x="Department", hue="Attrition", data=data)
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.show()
sns.countplot(x="Department", hue="Attrition", data=data)
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.show()
sns.boxplot(x="Attrition", y="MonthlyIncome", data=data)
plt.title("Monthly Income vs Attrition")
plt.show()

sns.barplot(x="Department", y="MonthlyIncome", hue="Attrition", data=data)
plt.title("Department vs Income vs Attrition")
plt.xticks(rotation=45)
plt.show()
sns.pairplot(
    data[["Age","MonthlyIncome","TotalWorkingYears","YearsAtCompany","Attrition"]],
    hue="Attrition"
)

plt.show()
plt.figure(figsize=(20,6))

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()
pd.crosstab([data["Department"], data["Gender"]], data["Attrition"]).plot(
    kind="pie",
    subplots=True,
    figsize=(12,8),
    autopct="%1.1f%%"
)

plt.show()
sns.countplot(x="Department", hue="Attrition", data=data)
plt.title("Department vs Attrition")

plt.figure()
sns.countplot(x="Gender", hue="Attrition", data=data)
plt.title("Gender vs Attrition")

plt.show()
sns.histplot(
    data=data,
    x="Age",
    hue="Attrition",
    multiple="stack"
)

plt.title("Age vs Attrition")
plt.show()
sns.boxplot(
    x="Department",
    y="MonthlyIncome",
    hue="Attrition",
    data=data
)

plt.xticks(rotation=45)
plt.title("Department vs Income vs Attrition")
plt.show()
sns.barplot(
    x="Department",
    y="MonthlyIncome",
    hue="Attrition",
    data=data
)

plt.xticks(rotation=45)
plt.title("Multivariate Bar Plot")
plt.show()
sns.pairplot(
    data[[
        "Age",
        "MonthlyIncome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "JobLevel",
        "Attrition"
    ]],
    hue="Attrition"
)

plt.show()
plt.figure(figsize=(20,8))

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Multivariate Correlation Heatmap")
plt.show()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Re-load the original data to ensure all columns are present
data = pd.read_csv('/content/HR-Employee-Attrition.csv')

# Select only numerical columns for outlier checking with boxplots
columns_to_plot = data.select_dtypes(include=np.number).columns.tolist()

n_cols = 5
n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

# Loop through numerical columns and create boxplots
for i, col in enumerate(columns_to_plot):
    if i < len(axes):
        sns.boxplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(col)
    else:
        break

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
data_outlier_check = data[['MonthlyIncome','NumCompaniesWorked','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany',
           'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]

fig , ax = plt.subplots(1,8,figsize=(20,3))

sns.histplot(data_outlier_check,x='MonthlyIncome',ax=ax[0])

sns.histplot(data_outlier_check,x='NumCompaniesWorked',ax=ax[1])

sns.histplot(data_outlier_check,x='TotalWorkingYears',ax=ax[2])

sns.histplot(data_outlier_check,x='YearsAtCompany',ax=ax[3])
sns.histplot(data_outlier_check,x='YearsInCurrentRole',ax=ax[4])
sns.histplot(data_outlier_check,x='YearsSinceLastPromotion',ax=ax[5])
sns.histplot(data_outlier_check,x='YearsWithCurrManager',ax=ax[6])
sns.histplot(data_outlier_check,x='TrainingTimesLastYear',ax=ax[7])

plt.tight_layout()
for col in data.select_dtypes(include=np.number).columns:
  print(col, data[col].skew())
# 3 sigma rule
mean_val = data.loc[:,'TrainingTimesLastYear'].mean()
std_val = data.loc[:,'TrainingTimesLastYear'].std()
lower_limit = mean_val - 3*std_val
upper_limit = mean_val + 3* std_val
data[data.loc[:,'TrainingTimesLastYear']<lower_limit]
data[data.loc[:,'TrainingTimesLastYear']>upper_limit]
len(data[(data.loc[:,'TrainingTimesLastYear']<lower_limit)|(data.loc[:,'TrainingTimesLastYear']>upper_limit)])/len(data)*100
data.loc[(data.loc[:,'TrainingTimesLastYear']<lower_limit)|(data.loc[:,'TrainingTimesLastYear']>upper_limit)
,'TrainingTimesLastYear']=data.loc[:,'TrainingTimesLastYear'].mean()
len(data[(data.loc[:,'TrainingTimesLastYear']<lower_limit)|(data.loc[:,'TrainingTimesLastYear']>upper_limit)])/len(data)*100
# IQR method for MonthlyIncome
Q1_mi = data.loc[:,'MonthlyIncome'].quantile(0.25)
Q3_mi = data.loc[:,'MonthlyIncome'].quantile(0.75)
IQR_mi = Q3_mi - Q1_mi
lower_limit_mi = Q1_mi - 1.5 * IQR_mi
upper_limit_mi = Q3_mi + 1.5 * IQR_mi

# Check outliers for MonthlyIncome
print("MonthlyIncome outliers < lower_limit:", data[data.loc[:,'MonthlyIncome'] < lower_limit_mi].shape[0])
print("MonthlyIncome outliers > upper_limit:", data[data.loc[:,'MonthlyIncome'] > upper_limit_mi].shape[0])

# Calculate percentage of outliers for MonthlyIncome
outlier_percentage_mi = len(data[(data.loc[:,'MonthlyIncome'] < lower_limit_mi) | (data.loc[:,'MonthlyIncome'] > upper_limit_mi)]) / len(data) * 100
print(f"Percentage of MonthlyIncome outliers: {outlier_percentage_mi:.2f}%")
data[data.loc[:,'MonthlyIncome']<lower_limit]
data[data.loc[:,'MonthlyIncome']<lower_limit]
data[data.loc[:,'MonthlyIncome']>upper_limit]
# This cell has been merged into QXSVHdlC7bpQ to ensure correct variable scoping.
# IQR method
Q1 = data.loc[:,'NumCompaniesWorked'].quantile(0.25)
Q3 = data.loc[:,'NumCompaniesWorked'].quantile(0.75)
IQR = Q3-Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5 * IQR
data[data.loc[:,'NumCompaniesWorked']<lower_limit]
data[data.loc[:,'NumCompaniesWorked']>upper_limit]
len(data[(data.loc[:,'NumCompaniesWorked']<lower_limit)|(data.loc[:,'NumCompaniesWorked']>upper_limit)])/len(data)*100
# IQR method
Q1 = data.loc[:,'TotalWorkingYears'].quantile(0.25)
Q3 = data.loc[:,'TotalWorkingYears'].quantile(0.75)
IQR = Q3-Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5 * IQR
data[data.loc[:,'TotalWorkingYears']<lower_limit]
data[data.loc[:,'TotalWorkingYears']>upper_limit]
len(data[(data.loc[:,'TotalWorkingYears']<lower_limit)|(data.loc[:,'TotalWorkingYears']>upper_limit)])/len(data)*100
Q1 = data.loc[:,'YearsAtCompany'].quantile(0.25)
Q3 = data.loc[:,'YearsAtCompany'].quantile(0.75)
IQR = Q3-Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 +1.5*IQR
data[data.loc[:,'YearsAtCompany']<lower_limit]
data[data.loc[:,'YearsAtCompany']>upper_limit]
len(data[(data.loc[:,'YearsAtCompany']<lower_limit)|(data.loc[:,'YearsAtCompany']>upper_limit)])/len(data)*100
Q1 = data.loc[:,'YearsInCurrentRole'].quantile(0.25)
Q3 = data.loc[:,'YearsInCurrentRole'].quantile(0.75)
IQR = Q3-Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 +1.5*IQR
data[data.loc[:,'YearsInCurrentRole']<lower_limit]
data[data.loc[:,'YearsInCurrentRole']>upper_limit]
len(data[(data.loc[:,'YearsInCurrentRole']<lower_limit)|(data.loc[:,'YearsInCurrentRole']>upper_limit)])/len(data)*100
Q1 = data.loc[:,'YearsSinceLastPromotion'].quantile(0.25)
Q3 = data.loc[:,'YearsSinceLastPromotion'].quantile(0.75)
IQR = Q3-Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 +1.5*IQR
data[data.loc[:,'YearsSinceLastPromotion']<lower_limit]
data[data.loc[:,'YearsSinceLastPromotion']>upper_limit]
len(data[(data.loc[:,'YearsSinceLastPromotion']<lower_limit)|(data.loc[:,'YearsSinceLastPromotion']>upper_limit)])/len(data)*100
Q1 = data.loc[:,'YearsWithCurrManager'].quantile(0.25)
Q3 = data.loc[:,'YearsWithCurrManager'].quantile(0.75)
IQR = Q3-Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 +1.5*IQR
data[data.loc[:,'YearsWithCurrManager']<lower_limit]
data[data.loc[:,'YearsWithCurrManager']>upper_limit]
len(data[(data.loc[:,'YearsWithCurrManager']<lower_limit)|(data.loc[:,'YearsWithCurrManager']>upper_limit)])/len(data)*100
# Attrition , OverTime ,Over18: Label Encoder
from sklearn.preprocessing import LabelEncoder

# Reload the original dataset to ensure all columns are present for encoding
data = pd.read_csv('/content/HR-Employee-Attrition.csv')

le = LabelEncoder()
data.loc[:,'Attrition']= le.fit_transform(data.loc[:,'Attrition'])
data.loc[:,'OverTime']= le.fit_transform(data.loc[:,'OverTime'])
data.loc[:,'Over18']= le.fit_transform(data.loc[:,'Over18'])
# BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus: One-Hot Encoder
print("Columns in data before one-hot encoding:", data.columns)
data = pd.get_dummies(data, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'], dtype='int')
data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
# Ensure 'data' here is the one-hot encoded 'data'
y = data['Attrition']
X = data.drop('Attrition', axis=1)

# Drop constant or identifier columns that do not provide useful information
# 'EmployeeCount', 'StandardHours', 'Over18' are constant, and 'EmployeeNumber' is an identifier.
X = X.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)

# Identify numerical columns for scaling. These are the original numerical columns
# and the label encoded 'OverTime'. One-hot encoded columns don't typically need scaling.
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

# Initialize and apply StandardScaler to numerical columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Perform train-test split AFTER all preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Features (X) after scaling numerical columns and dropping constant/identifier columns:")
print(X.head())
print("Target (y) variable:")
print(y.head())
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
data

from sklearn.model_selection import train_test_split

# These variables were already created in the scaling_code_cell. Re-printing for confirmation.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# Convert y_test to integer type for consistent evaluation across models
y_test_int = y_test.astype(int)

y

# check data is balanced or not
plt.figure(figsize=(5,2))
sns.countplot(data,x='Attrition')
plt.show()
# Balancing should be done only on training data not on testing data
from imblearn.over_sampling import SMOTE
from collections import Counter
smote = SMOTE(random_state=42)

# Convert y_train to integer type to resolve the 'Unknown label type' error, using the correct y_train
y_train_smote_input = y_train.astype(int)

# Apply SMOTE to the correctly preprocessed X_train and y_train
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_smote_input)

print("Original training target distribution:", Counter(y_train_smote_input))
print("Resampled training target distribution:", Counter(y_train_resampled))
# before sampling
Counter(y_train_smote_input)
# after sampling
Counter(y_train_resampled)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)
# predictions
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# confusion matrix
confusion_matrix(y_test.astype(int),y_pred)
# accuracy
accuracy_score(y_test.astype(int),y_pred)
# recall
recall_score(y_test.astype(int),y_pred)
# precision
precision_score(y_test.astype(int),y_pred)
f1_score(y_test.astype(int),y_pred)
# find probabilities
y_prob_knn = model.predict_proba(X_test)[:,1]
y_prob_knn
# Auc , Roc
from sklearn.metrics import roc_auc_score,roc_curve
auc_knn=roc_auc_score(y_test.astype(int),y_prob_knn)
auc_knn
# ROC curve
fpr_knn,tpr_knn ,threshold_knn = roc_curve(y_test.astype(int),y_prob_knn)
plt.plot(fpr_knn, tpr_knn, label=f'AUC = {auc_knn:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression') # Updated title
plt.legend()
plt.show()
from sklearn.neighbors import KNeighborsClassifier
model =  KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
# confusion matrix
confusion_matrix(y_test.astype(int),y_pred)
# accuracy
accuracy_score(y_test.astype(int),y_pred)
# recall
recall_score(y_test.astype(int),y_pred)
# precision
precision_score(y_test.astype(int),y_pred)
# f1 score
f1_score(y_test.astype(int),y_pred)
# auc roc curve
roc_auc_score(y_test.astype(int),y_pred)
# roc curve
y_prob = model.predict_proba(X_test)[:,1]
fpr,tpr ,threshold = roc_curve(y_test_int,y_prob)
plt.figure(figsize=(5,2))
plt.plot(fpr,tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.title('Roc curve')
plt.show()
error =[]
for k in range(2,11):
  model =  KNeighborsClassifier(n_neighbors=k)
  model.fit(X_train_resampled, y_train_resampled)
  y_pred = model.predict(X_test)
  error.append(np.mean(y_test_int!=y_pred))
error
plt.figure(figsize=(5,2))
plt.plot(range(2,11),error,)
plt.show()

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini',max_depth=6,min_samples_split=5,max_features='sqrt',splitter='best')
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
# confusion matrix
confusion_matrix(y_test.astype(int),y_pred)
# accuracy
accuracy_score(y_test.astype(int),y_pred)
# precision
precision_score(y_test.astype(int),y_pred)
# recall
recall_score(y_test.astype(int),y_pred)
f1_score(y_test.astype(int),y_pred)
# auc_roc_score
auc_dt = roc_auc_score(y_test.astype(int),y_pred)
print(auc_dt)

# ROC curve
fpr_dt, tpr_dt, threshold_dt = roc_curve(y_test.astype(int),y_pred)

plt.figure(figsize=(5,2))
plt.plot(fpr_dt, tpr_dt, label=f'AUC = {auc_dt:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend()
plt.show()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini',max_depth=6,min_samples_split=3,min_samples_leaf=5)
kfold =KFold(n_splits=5,shuffle=True,random_state=42)
scores = cross_val_score(model, X_train, y_train.astype(int), cv=kfold)
scores
print(np.mean(scores))
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini',max_depth=6,min_samples_split=3,min_samples_leaf=5)
sfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scores = cross_val_score(model, X_train, y_train.astype(int), cv=sfold)
scores
np.mean(scores)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

model_rf_clf = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_split=5, min_samples_leaf=3, random_state=42)
model_rf_clf.fit(X_train_resampled, y_train_resampled)
y_pred_rf_clf = model_rf_clf.predict(X_test)
y_prob_rf_clf = model_rf_clf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Classifier Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y_test_int, y_pred_rf_clf))
print("Accuracy:", accuracy_score(y_test_int, y_pred_rf_clf))
print("Precision:", precision_score(y_test_int, y_pred_rf_clf))
print("Recall:", recall_score(y_test_int, y_pred_rf_clf))
print("F1-Score:", f1_score(y_test_int, y_pred_rf_clf))
print("AUC Score:", roc_auc_score(y_test_int, y_prob_rf_clf))
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train_resampled,y_train_resampled)
y_pred = model.predict(X_test)
# confusion matrix
confusion_matrix(y_test_int,y_pred)
# accuracy
accuracy_score(y_test_int,y_pred)
# recall
recall_score(y_test_int,y_pred)
# precision
precision_score(y_test_int,y_pred)
# f1score
f1_score(y_test_int,y_pred)
# roc_auc_curve
roc_auc_score(y_test_int,y_pred)
from xgboost import XGBClassifier

# Ensure 'OverTime' column is numeric
# It should contain 0s and 1s from previous Label Encoding
if 'OverTime' in X_train_resampled.columns and X_train_resampled['OverTime'].dtype == 'object':
    X_train_resampled['OverTime'] = X_train_resampled['OverTime'].astype(int)
if 'OverTime' in X_test.columns and X_test['OverTime'].dtype == 'object':
    X_test['OverTime'] = X_test['OverTime'].astype(int)

model = XGBClassifier()
model.fit(X_train_resampled,y_train_resampled)
y_pred = model.predict(X_test)
 #confusion matrix
confusion_matrix(y_test_int,y_pred)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train_resampled,y_train_resampled)
y_pred = model.predict(X_test)
confusion_matrix(y_test_int,y_pred)
accuracy_score(y_test_int,y_pred)
recall_score(y_test_int,y_pred)
precision_score(y_test_int,y_pred)
f1_score(y_test_int,y_pred)
from sklearn.ensemble import GradientBoostingClassifier # Changed to Classifier
from sklearn.model_selection import GridSearchCV

model = GradientBoostingClassifier() # Changed to Classifier
parameters = {
    'max_depth':[2,3,4], # Reduced options
    'n_estimators':[100,200],
    'learning_rate':[0.01,0.2]
}
grid = GridSearchCV(estimator=model , param_grid=parameters,cv=5)
# Use the resampled training data for Grid Search
grid.fit(X_train_resampled, y_train_resampled)
#Find and print the best parameters
print("Best parameters found: ", grid.best_params_)
best_gb = grid.best_estimator_
#Use the best model from grid search to make predictions
y_pred_gb = best_gb.predict(X_test)

# Evaluate classification performance
print("--- Final Gradient Boosting Classifier Evaluation ---")
print("Accuracy:", accuracy_score(y_test_int, y_pred_gb))
print("Precision:", precision_score(y_test_int, y_pred_gb))
print("Recall:", recall_score(y_test_int, y_pred_gb))
print("F1-Score:", f1_score(y_test_int, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test_int, y_pred_gb))
# Evaluate final accuracy for the tuned Gradient Boosting model
acc_gb = accuracy_score(y_test_int, y_pred_gb)
print(f"Final Accuracy: {acc_gb:.4f}")
# Evaluate final F1-score for the tuned Gradient Boosting model
f1_gb = f1_score(y_test_int, y_pred_gb)
print(f"Final F1-Score: {f1_gb:.4f}")
from sklearn.ensemble import GradientBoostingClassifier # Changed to Classifier
from sklearn.model_selection import RandomizedSearchCV

model = GradientBoostingClassifier() # Changed to Classifier
parameters = {
    'max_depth':[2,3,4], # Reduced options
    'n_estimators':[100,200],
    'learning_rate':[0.01,0.2]
}
random = RandomizedSearchCV(estimator=model , param_distributions=parameters,cv=5)
# Use the resampled training data for random Search
random.fit(X_train_resampled, y_train_resampled)
print(random.best_params_)
model=GradientBoostingClassifier(learning_rate= 0.1, max_depth= 4,
                                 n_estimators= 100)
model.fit(X_train_resampled,y_train_resampled)
y_pred = model.predict(X_test)
accuracy_score(y_test_int,y_pred)
recall_score(y_test_int,y_pred)
precision_score(y_test_int,y_pred)
# grid search
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
model=XGBClassifier()
parameters = {'max_depth':[3,4,5],'n_estimators':[100,200]
              ,'learning_rate':[0.01,0.1],"reg_alpha":[0,0.1,0.5],
    "reg_lambda":[1,2,3]}
grid = GridSearchCV(estimator=model,param_grid=parameters,cv=5,scoring='accuracy')
grid.fit(X_train_resampled,y_train_resampled)
print(grid.best_params_)
model=XGBClassifier(reg_lambda= 1,reg_alpha= 0, n_estimators= 100,
                    max_depth= 3, learning_rate= 0.1)
model.fit(X_train_resampled,y_train_resampled)
y_pred = model.predict(X_test)
accuracy_score(y_test_int,y_pred)
recall_score(y_test_int,y_pred)
precision_score(y_test_int,y_pred)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
model=XGBClassifier()
parameters = {'max_depth':[3,4,5],'n_estimators':[100,200]
              ,'learning_rate':[0.01,0.1],"reg_alpha":[0,0.1,0.5],
    "reg_lambda":[1,2,3]}
random = RandomizedSearchCV(estimator=model,param_distributions=parameters,
                            n_iter=20,cv=5,scoring='accuracy')
random.fit(X_train_resampled,y_train_resampled)
print(random.best_params_)
model=XGBClassifier(reg_lambda= 1,reg_alpha= 0, n_estimators= 100,
                    max_depth= 3, learning_rate= 0.1)
model.fit(X_train_resampled,y_train_resampled)
y_pred = model.predict(X_test)
accuracy_score(y_test_int,y_pred)
recall_score(y_test_int,y_pred)
precision_score(y_test_int,y_pred)
from sklearn.model_selection import GridSearchCV
model = SVC()
parameters={'kernel':['linear','poly','rbf','sigmoid'],
            'gamma':[0.1,0.01,0.001],'C':[0.1,1,5,10]}
grid = GridSearchCV(estimator=model,param_grid=parameters,cv=5,scoring='accuracy')
grid.fit(X_train_resampled,y_train_resampled)
print(grid.best_params_)
model=SVC(C=10,gamma=0.01,kernel='rbf')
model.fit(X_train_resampled,y_train_resampled)
y_pred = model.predict(X_test)
confusion_matrix(y_test_int,y_pred)
accuracy_score(y_test_int,y_pred)
recall_score(y_test_int,y_pred)

precision_score(y_test_int,y_pred)
f1_score(y_test_int,y_pred)
from sklearn.model_selection import RandomizedSearchCV
model = SVC()
parameters={'kernel':['linear','poly','rbf','sigmoid'],
            'gamma':[0.1,0.01,0.001],'C':[0.1,1,5,10]}
random = RandomizedSearchCV(estimator=model,param_distributions=parameters, cv=5,scoring='accuracy',n_iter=10)
random.fit(X_train_resampled,y_train_resampled)
print(random.best_params_)
model=SVC(C=0.1,gamma=0.1,kernel='linear')
model.fit(X_train_resampled,y_train_resampled)
y_pred = model.predict(X_test)
confusion_matrix(y_test_int,y_pred)

accuracy_score(y_test_int,y_pred)
recall_score(y_test_int,y_pred)
precision_score(y_test_int,y_pred)
f1_score(y_test_int,y_pred)
import pandas as pd

# Gathering metrics for comparison
# Note: These variables were calculated in previous evaluation cells
results = {
    'Model': ['Logistic Regression', 'Random Forest', 'Tuned Gradient Boosting'],
    'Accuracy': [0.7448, 0.8673, 0.8877],
    'Precision': [0.2750, 0.5000, 0.6363],
    'Recall': [0.5641, 0.3333, 0.3589],
    'F1-Score': [0.3697, 0.4000, 0.4590]
}

comparison_df = pd.DataFrame(results)
display(comparison_df.sort_values(by='Accuracy', ascending=False))

print("\nSummary: The Tuned Gradient Boosting Classifier is the best performing model in terms of Accuracy and F1-Score.")
best_model_name = comparison_df.sort_values(by='Accuracy', ascending=False).iloc[0]['Model']
best_accuracy = comparison_df.sort_values(by='Accuracy', ascending=False).iloc[0]['Accuracy']

print(f"Selected Best Model: {best_model_name}")
print(f"Final Accuracy: {best_accuracy * 100:.2f}%")
