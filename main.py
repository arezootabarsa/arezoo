import fuzz
import graphviz
import matplotlib
import pandas as pd
import numpy as np
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import roc_curve, auc, silhouette_score, precision_score, recall_score, f1_score
from pyexpat import model

from sklearn import set_config
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils import graph

df = pd.read_csv("C:/Users/TBS/Desktop/survey lung cancer.csv")
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

colors = ['#011f4b', '#03396c', '#005b96', '#6497b1', '#b3cde0']
sns.set_palette(sns.color_palette(colors))
pd.set_option('display.max_columns', None)
print(df.head())
print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df.drop_duplicates(inplace=True))
print(df.shape)
gender = df['GENDER'].value_counts()
print(gender.values)
print(gender.index)
con_col = ['AGE']
cat_col = []
for i in df.columns:
    if i != 'AGE':
        cat_col.append(i)
fig,ax = plt.subplots(1,3,figsize=(20,6))
sns.histplot(df['AGE'],ax=ax[0])
sns.histplot(data=df, x='AGE', ax=ax[1], hue='LUNG_CANCER', kde=True)
sns.boxplot(x=df['LUNG_CANCER'], y=df['AGE'], ax=ax[2])
plt.suptitle("Visualizing AGE column", size=20)
plt.show()
fig,ax = plt.subplots(15,3, figsize=(30,90))
for index,i in enumerate(cat_col):
    sns.boxplot(x=df[i], y=df['AGE'], ax=ax[index,0])
    sns.boxplot(x=df[i], y=df['AGE'], ax=ax[index,1],hue=df['LUNG_CANCER'])
    sns.violinplot(x=df[i], y=df['AGE'], ax=ax[index,2])
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.suptitle("Visualizing AGE vs Categorical Columns", fontsize=50)
plt.savefig('plot3')

encoder = LabelEncoder()
df['LUNG_CANCER']=encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER']=encoder.fit_transform(df['GENDER'])
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot=True, linewidth=0.5, fmt='0.2f')
plt.savefig('plot4')


X = df.drop(['LUNG_CANCER'], axis=1)
y = df['LUNG_CANCER']

for i in X.columns[2:]:
    temp = []
    for j in X[i]:
        temp.append(j-1)
    X[i] = temp
X.head()

print(X.to_string(index=False))
print(X)

from imblearn.over_sampling import RandomOverSampler
X_over, y_over = RandomOverSampler().fit_resample(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X_over, y_over, random_state=42, stratify=y_over)
print(f'Train shape : {X_train.shape}\nTest shape: {X_test.shape}')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train['AGE'] = scaler.fit_transform(X_train[['AGE']])
X_test['AGE'] = scaler.transform(X_test[['AGE']])
print(X_train.head())

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
rcv = RandomizedSearchCV(SVC(kernel='sigmoid'), param_grid, cv=5)
rcv.fit(X_train,y_train)
y_pred_svc = rcv.predict(X_test)
confusion_svc = confusion_matrix(y_test, rcv.predict(X_test))
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_svc, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('plot6')
print(classification_report(y_test, y_pred_svc))
print(f'\nBest Parameters of SVC model is : {rcv.best_params_}\n')
print('model accuricy:{0:0.7f}'.
format(accuracy_score(y_test,y_pred_svc)))
def plot_roc_curve(y_true, y_probs, algorithm_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{algorithm_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {algorithm_name}')
    plt.legend(loc="lower right")

y_probs_svc = rcv.decision_function(X_test)
plot_roc_curve(y_test, y_probs_svc, 'SVM (sigmoid Kernel)')
plt.savefig('plot11')
plt.show()

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
rcv = RandomizedSearchCV(SVC(kernel='poly', degree=3), param_grid, cv=5)
rcv.fit(X_train, y_train)
y_pred_svc = rcv.predict(X_test)
confusion_svc = confusion_matrix(y_test, y_pred_svc)
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_svc, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('plot7')
plt.show()
print(classification_report(y_test, y_pred_svc))
print(f'\nBest Parameters of SVC model is : {rcv.best_params_}\n')
print('model accuracy: {:.7f}'.format(accuracy_score(y_test, y_pred_svc)))
y_probs_svc = rcv.decision_function(X_test)
plot_roc_curve(y_test, y_probs_svc, 'SVM (Poly Kernel)')
plt.savefig('plot10')
plt.show()

#
#
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
rcv = RandomizedSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
rcv.fit(X_train,y_train)
y_pred_svc = rcv.predict(X_test)
confusion_svc = confusion_matrix(y_test, rcv.predict(X_test))
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_svc, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('plot8')
print(classification_report(y_test, y_pred_svc))
print(f'\nBest Parameters of SVC model is : {rcv.best_params_}\n')
print('model accuricy:{0:0.7f}'.
format(accuracy_score(y_test,y_pred_svc)))
y_probs_svc = rcv.decision_function(X_test)
plot_roc_curve(y_test, y_probs_svc, 'SVM (rbf Kernel)')
plt.savefig('plot12')
plt.show()
#
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
rcv = RandomizedSearchCV(SVC(kernel='linear'), param_grid, cv=5)
rcv.fit(X_train,y_train)
y_pred_svc = rcv.predict(X_test)
confusion_svc = confusion_matrix(y_test, rcv.predict(X_test))
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_svc, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('plot9')
print(classification_report(y_test, y_pred_svc))
print(f'\nBest Parameters of SVC model is : {rcv.best_params_}\n')
print('model accuricy:{0:0.7f}'.
format(accuracy_score(y_test,y_pred_svc)))
y_probs_svc = rcv.decision_function(X_test)
plot_roc_curve(y_test, y_probs_svc, 'SVM (linear Kernel)')
plt.savefig('plot13')
plt.show()
#
# param_grid={'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.001,0.01,0.1,1,10,100]}
# rcv=RandomizedSearchCV(SVC(),param_grid,cv=5)
# rcv.fit(X_train,y_train)
# y_pred_svc=rcv.predict(X_test)
# confusion_svc=confusion_matrix(y_test,rcv.predict(X_test))
# plt.figure(figsize=(8,8))
# sns.heatmap(confusion_svc,annot=True)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.savefig('plot10')
# print(classification_report(y_test,y_pred_svc))
# print(f'\nBest Parameters of SVC model is : {rcv.best_params_}\n')
# print('model accuricy:{0:0.7f}'.
# format(accuracy_score(y_test,y_pred_svc)))
#
#
# age = df['AGE']
# age_array = age.values.reshape(-1, 1)
# scaler = MinMaxScaler()
# normalized_age = scaler.fit_transform(age_array)
# normalized_age_series = pd.Series(normalized_age.flatten())
# df['AGE'] = normalized_age_series
# print(df)
#
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training Accuracy of Decision Tree Model is {dtc_train_acc}")
print(f"Test Accuracy of Decision Tree Model is {dtc_test_acc}")
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plot_tree(dtc, feature_names=X.columns, class_names=["NO", "YES"], filled=True, rounded=True)
plt.savefig('plot28',dpi=700)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


dtc_id3 = DecisionTreeClassifier(criterion='entropy')
dtc_id3.fit(X_train, y_train)


y_pred_id3 = dtc_id3.predict(X_test)


id3_train_acc = accuracy_score(y_train, dtc_id3.predict(X_train))
id3_test_acc = accuracy_score(y_test, y_pred_id3)


print(f"Training Accuracy of ID3 Decision Tree Model is {id3_train_acc}")
print(f"Test Accuracy of ID3 Decision Tree Model is {id3_test_acc}")


confusion_matrix(y_test, y_pred_id3)
plt.figure(figsize=(15, 10))
plot_tree(dtc, feature_names=X.columns, class_names=["NO", "YES"], filled=True, rounded=True)
plt.savefig('plot33',dpi=700)
y_probs_id3 = dtc_id3.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_probs_id3, 'Decision Tree (id3)')
plt.savefig('plot14')
plt.show()



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred_RandomForest = rfc.predict(X_test)

rfc_train_acc = accuracy_score(y_train, rfc.predict(X_train))
rfc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training Accuracy of Random Forest Model is: {rfc_train_acc}")
print(f"Test Accuracy of Random Forest Model is: {rfc_test_acc}")

confusion_matrix_result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix_result)
plt.figure(figsize=(15, 10))
plot_tree(dtc, feature_names=X.columns, class_names=["NO", "YES"], filled=True, rounded=True)
plt.savefig('plot30',dpi=700)
y_probs_RandomForest = rfc.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_probs_RandomForest, 'Decision Tree (RandomForest)')
plt.savefig('plot15')
plt.show()

#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
#
#
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
#
#
# y_pred = dtc.predict(X_test)
#
#
# dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
# dtc_test_acc = accuracy_score(y_test, y_pred)
#
#
# print(f"Training Accuracy of Decision Tree Model is {dtc_train_acc}")
# print(f"Test Accuracy of Decision Tree Model is {dtc_test_acc}")
#
#
# confusion_matrix(y_test, y_pred)
#
#
# plt.figure(figsize=(15, 10))
# plot_tree(dtc, feature_names=X.columns, class_names=["NO", "YES"], filled=True, rounded=True)
# plt.savefig('plot44',dpi=700)
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
#
#
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
#
#
# y_pred = dtc.predict(X_test)
#
#
# dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
# dtc_test_acc = accuracy_score(y_test, y_pred)
#
#
# print(f"Training Accuracy of Decision Tree Model is {dtc_train_acc}")
# print(f"Test Accuracy of Decision Tree Model is {dtc_test_acc}")
#
#
# confusion_matrix(y_test, y_pred)
#
# plt.figure(figsize=(15, 10))
# plot_tree(dtc, feature_names=X.columns, class_names=["NO", "YES"], filled=True, rounded=True)
# plt.savefig('plot17',dpi=700)
#

# z=pd.DataFrame(df,columns=['AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','CHEST PAIN','SWALLOWING DIFFICULTY','WHEEZING','COUGHING','SHORTNESS OF BREATH'])
# y= df["LUNG_CANCER"]
# import sklearn.cluster as cluster
# plt.figure(figsize=(8,7))
# Kmean = cluster.KMeans(n_clusters=10, n_init=10)
# Kmean = Kmean.fit(z)
# df['Cluster'] = Kmean.labels_
# plt.scatter(z['AGE'], z['SMOKING'] ,c=df['Cluster'], cmap='viridis')
# plt.title('K-Means Clustering')
# plt.xlabel('AGE')
# plt.ylabel('SMOKING')
# Sse= Kmean.inertia_
# print(f"SSE K-mean:{Sse}")
# plt.savefig('plot30', dpi=700)
# fpr, tpr, thresholds = roc_curve(y, df['Cluster'])
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Khushe bandi K-mean (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.savefig('plot16', dpi=700)
# plt.show()
#
# from sklearn.cluster import DBSCAN
# import skfuzzy as fuzz
# LUNG_CANCER=df["LUNG_CANCER"]
# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#     z.T, 10, 2, error=0.005, maxiter=100, init=None)
#
# sse = 0.0
# for i in range(len(z)):
#     for j in range(10):
#         sse += u[j, i] ** 2 * ((z.iloc[i] - cntr[j])**2).sum()
# cluster_membership = pd.DataFrame(u.T, columns=[f'Cluster_{i+1}' for i in range(10)])
# df['Cluster'] = cluster_membership.idxmax(axis=1)
# binary_labels = (cluster_membership['Cluster_1'] > 0.5).astype(int)
# fpr, tpr, thresholds = roc_curve(LUNG_CANCER, binary_labels)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.savefig('plot17', dpi=700)
# plt.show()
# print(f"SSE Fuzzy C-Mean: {sse}")
# cluster_membership = pd.DataFrame(u.T, columns=[f'Cluster_{i+1}' for i in range(10)])#c
# df['Cluster'] = cluster_membership.idxmax(axis=1)
# plt.scatter(z['AGE'], z['SMOKING'], c=df['Cluster'].astype('category').cat.codes, cmap='viridis')
# plt.title('Fuzzy C-Means Clustering')
# plt.xlabel('AGE')
# plt.ylabel('SMOKING')
# plt.savefig('plot50', dpi=700)
# #
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(z)
# epsilon = 0.5
# min_samples = 10
# dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
# dbscan.fit(X_scaled)
# df['dbscan_cluster'] = dbscan.labels_
# plt.scatter(z['AGE'], z['SMOKING'], c=df['dbscan_cluster'], cmap='viridis')
# plt.title('DBSCAN Clustering')
# plt.xlabel('AGE')
# plt.ylabel('SMOKING')
# plt.savefig('plot35', dpi=700)
# fpr, tpr, thresholds = roc_curve(LUNG_CANCER, binary_labels)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.savefig('plot19', dpi=700)
# plt.show()

