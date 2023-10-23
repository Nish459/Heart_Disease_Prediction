# importing required libraries
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.impute import KNNImputer

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import f_classif
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from skrebate import ReliefF

# loading and reading the dataset
data1=pd.read_csv("heart.dat",sep=" ")
data1['num']= [0 if x==1 else 1 for x in data1['num']]
data2=pd.read_csv("processed.hungarian.data", sep=",")
data3=pd.read_csv("processed.switzerland.data", sep=",")
data4=pd.read_csv("processed.va.data", sep=",")
data5=pd.read_csv("processed.cleveland.data", sep=",")

df1=pd.concat([data1,data2],axis=0)
df2=pd.concat([data3,data4],axis=0)
df3=pd.concat([df1,df2],axis=0)
df=pd.concat([df3,data5],axis=0)
df['num']= [0 if x==0 else 1 for x in df['num']]

df = df.replace({'?': np.nan})

# Split the dataset into two parts - one with missing values and the other with non-missing values
df_missing = df[df.isnull().any(axis=1)]
df_non_missing = df.dropna()

# Normalize the dataset using MinMaxScaler
scaler = MinMaxScaler()
df_non_missing_scaled = pd.DataFrame(scaler.fit_transform(df_non_missing), columns=df_non_missing.columns)

# Apply KNN imputation using the KNNImputer method
imputer = KNNImputer(n_neighbors=5)
df_missing_imputed = pd.DataFrame(imputer.fit_transform(df_missing), columns=df_missing.columns)

#updating
df_missing_imputed.index = df_missing.index
df_cleaned = pd.concat([df_non_missing, df_missing_imputed], ignore_index=True)
df_cleaned = df_cleaned.sort_index()

columns_to_std = ['Age','Sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

# Calculate the mean and standard deviation of the selected columns
scaler = StandardScaler()

# fit and transform the selected columns only
df_cleaned[columns_to_std] = scaler.fit_transform(df_cleaned[columns_to_std])

features= ['Age','Sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
X= df_cleaned[features]
X= X.values
y= df_cleaned['num'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that selects the top 6 features using the Relief feature selection technique
pipeline = make_pipeline(
    SelectKBest(mutual_info_classif, k=6),
    KNeighborsClassifier(n_neighbors=3)
)
# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Print the selected features
selected_features = pipeline.named_steps['selectkbest'].get_support()
print("Selected features:", selected_features)

df_new = df_cleaned.drop(['Age','Sex','trestbps','fbs','restecg','slope','thal'], axis=1)

X = df_new.drop('num', axis=1)
y = df_new['num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

# fit the classifier to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
p=model.score(X_test,y_test)
print(p)
# print(f'Test accuracy: {accuracy:.2f}')
print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# heart_df=df


# # Renaming some of the columns 
# heart_df = heart_df.rename(columns={'num':'target'})
# print(heart_df.head())

# df = heart_df.drop(['Age','Sex','trestbps','fbs','restecg','slope','thal'], axis=1)
# # model building 

# #fixing our data in x and y. Here y contains target data and X contains rest all the features.
# x= df.drop(columns= 'target')
# y= df.target

# # splitting our dataset into training and testing for this we will use train_test_split library.
# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

# #feature scaling
# scaler= StandardScaler()
# x_train_scaler= scaler.fit_transform(x_train)
# x_test_scaler= scaler.fit_transform(x_test)

# # creating K-Nearest-Neighbor classifier
# model=RandomForestClassifier(n_estimators=100)
# model.fit(x_train_scaler, y_train)
# y_pred= model.predict(x_test_scaler)
# p = model.score(x_test_scaler,y_test)
# print(p)

# print('Classification Report\n', classification_report(y_test, y_pred))
# print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-knn-model.pkl'
pickle.dump(model, open(filename, 'wb'))

