import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from funzioni import *

import warnings
warnings.filterwarnings("ignore")


dataset = pd.read_csv("heart.csv")
print("Number of null data:\n",dataset.isnull().sum())
print("Number of duplicated data: ",dataset.duplicated().sum())
dataset.drop_duplicates(inplace=True)
print("Number of duplicated data: ",dataset.duplicated().sum())

categorical_data = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
numeric_data = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']



dataset['sex'] = dataset['sex'].replace({0: 'Female', 1: 'Male'})
categoricalDataVisualization(dataset,'sex','Sex (Gender)',[0,1])
dataset['sex'] = dataset['sex'].replace({'Female': 0, 'Male': 1})

dataset['cp'] = dataset['cp'].replace({0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non Anginal Pain', 3:'Asympotomatic'})
categoricalDataVisualization(dataset,'cp','Chest Pain',[0,1,2,3])
dataset['cp'] = dataset['cp'].replace({'Typical Angina': 0 , 'Atypical Angina': 1, 'Non Anginal Pain': 2, 'Asympotomatic': 3})

dataset['fbs'] = dataset['fbs'].replace({0: '<= 120 mg/dL', 1: '> 120 mg/dL'})
categoricalDataVisualization(dataset,'fbs','Fasting Blood Sugar',[0,1])
dataset['fbs'] = dataset['fbs'].replace({'<= 120 mg/dL': 0, '> 120 mg/dL': 1})

dataset['restecg'] = dataset['restecg'].replace({0: 'Normal', 1:'ST_T Wave Abnormality', 2:'Left Ventricular\n Hypertrophy'})
categoricalDataVisualization(dataset,'restecg','Resting ECG Results',[0,1,2])
dataset['restecg'] = dataset['restecg'].replace({'Normal': 0,'ST_T Wave Abnormality': 1, 'Left Ventricular\n Hypertrophy': 2})

dataset['exng'] = dataset['exng'].replace({0: 'No', 1: 'Yes'})
categoricalDataVisualization(dataset,'exng','Exercise-Induced Angina',[0,1])
dataset['exng'] = dataset['exng'].replace({'No': 0, 'Yes': 1})

dataset['slp'] = dataset['slp'].replace({0: 'Downsloping', 1: 'Flat', 2: 'UpslopingDiagnosis'})
categoricalDataVisualization(dataset,'slp','Slope of ST Segment',[0,1,2])
dataset['slp'] = dataset['slp'].replace({'Downsloping': 0,'Flat': 1,'UpslopingDiagnosis': 2})

categoricalDataVisualization(dataset,'caa','Number of Major Vessels Colored by Fluoroscopy',[0,1,2,3,4])

dataset['thall'] = dataset['thall'].replace({0: 'None', 1: 'FixedDefect', 2: 'ReversibleDefect', 3:'Thalassemia' })
categoricalDataVisualization(dataset,'thall','Thalassemia Type',[0,1,2,3])
dataset['thall'] = dataset['thall'].replace({'None': 0,'FixedDefect': 1,'ReversibleDefect': 2, 'Thalassemia':3})

dataset['output'] = dataset['output'].replace({0: 'No', 1: 'Yes'})
categoricalDataVisualization(dataset,'output','Heart Diseases',[0,1])
dataset['output'] = dataset['output'].replace({'No': 0, 'Yes': 1})

numericDataVisualization(dataset, 'age', 'Age')
numericDataVisualization(dataset, 'trtbps', 'Resting Blood Pressure')
numericDataVisualization(dataset, 'chol', 'Cholestoral')
numericDataVisualization(dataset, 'thalachh', 'Maximum Heart Rate')
numericDataVisualization(dataset, 'oldpeak', 'ST-Segment Depression')

labelFeature = ['Female','Male']
dataExploration(dataset, 'sex','Gender',labelFeature, [0,1])

labelFeature=['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
dataExploration(dataset, 'cp','Chest Pain',labelFeature, [0,1,2,3])

labelFeature=['<= 120 mg/dL', '> 120 mg/dL']
dataExploration(dataset, 'fbs','Fasting Blood Sugar',labelFeature, [0,1])

labelFeature=['Normal','ST_T Wave Abnormality','Left Ventricular Hypertrophy']
dataExploration(dataset, 'restecg','Resting ECG Results',labelFeature, [0,1,2])

labelFeature = ['No', 'Yes']
dataExploration(dataset, 'exng','Exercise-Induced Angina',labelFeature, [0,1])

labelFeature = ['Downsloping', 'Flat','UpslopingDiagnosis']
dataExploration(dataset,'slp','Slope of ST Segment',labelFeature,[0,1,2])

labelFeature = ['0','1','2','3','4']
dataExploration(dataset,'caa','Number of Major Vessels Colored by Fluoroscopy',labelFeature,[0,1,2,3,4])

labelFeature = ['None','FixedDefect','ReversibleDefect','Thalassemia']
dataExploration(dataset,'thall','Thalassemia Type',labelFeature,[0,1,2,3])

dataExplorationNumeric(dataset, dataset.chol, 'Cholestoral')
dataExplorationNumeric(dataset, dataset.trtbps, 'Resting Blood Pressure')
dataExplorationNumeric(dataset, dataset.thalachh, 'Maximum Heart Rate')

#heatmap
plt.figure(figsize= (15,6))
sns.heatmap(dataset.corr(), annot = True)
plt.show()

'''--------------------------------------------------------------------------------------------------------------------------'''

#Standardization
data = dataset.copy()
data_df = pd.get_dummies(data, columns=categorical_data[:-1], dtype=np.uint8)

scaler=StandardScaler()
data_df[numeric_data[:-1]] = scaler.fit_transform(data[numeric_data[:-1]])

X=data_df.drop(["output"],axis=1)
y=data_df ["output"]

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''--------------------------------------------------------------------------------------------------------------------------'''

classifiers = []
model_result_accuracy = []
model_result_precision = []
model_result_recall = []
model_result_f1 = []

model_result_accuracy_hyperparameters=[]
model_result_f1_hyperparameters=[]
model_result_precision_hyperparameters=[]
model_result_recall_hyperparameters=[]

'''--------------------------------------------------------------------------------------------------------------------------'''

#Logistic Regression
classifiers.append("LogisticRegression")
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

log_reg_accuracy = accuracy_score(y_test, y_pred)
log_reg_recall = recall_score(y_test,y_pred)
log_reg_precision = precision_score(y_test,y_pred)
log_reg_f1 = f1_score(y_test,y_pred)

model_result_accuracy.append(log_reg_accuracy)
model_result_f1.append(log_reg_f1)
model_result_precision.append(log_reg_precision)
model_result_recall.append(log_reg_recall)

print('Logistic Regression')

print('ACCURACY : ', log_reg_accuracy)
print('F1       : ', log_reg_f1)
print('PRECISION: ', log_reg_precision)
print('RECALL   : ', log_reg_recall)

confusionMatrix (y_test, y_pred,"Logistic Regression")
rocCurve(log_reg,"Logistic Regression", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''

#K-Nearest Neighbors
classifiers.append("KNearestNeighbors")
k_values = [i for i in range (1,20)]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv = 5)
    scores.append(np.mean(score))
    
plt.plot(k_values, scores, color='g')
plt.xticks(ticks=k_values, labels=k_values)
plt.title("Number Of Neighbors")
plt.grid()
plt.show()

best_index = np.argmax(scores)
best_k = k_values[best_index]

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred)
knn_precision = precision_score(y_test, y_pred)
knn_recall = recall_score(y_test, y_pred)
knn_f1 = f1_score(y_test, y_pred)

model_result_accuracy.append(knn_accuracy)
model_result_f1.append(knn_f1)
model_result_precision.append(knn_precision)
model_result_recall.append(knn_recall)

print('K-Nearest Neighbors')

print('ACCURACY : ', knn_accuracy)
print('F1       : ', knn_f1)
print('PRECISION: ', knn_precision)
print('RECALL   : ', knn_recall)

confusionMatrix (y_test, y_pred, "K-Nearest Neighbors")
rocCurve(knn,"K-Nearest Neighbors", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''

#Support Vector Machines
classifiers.append("SupportVectorMachines")
svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test,y_pred)
svm_recall = recall_score (y_test,y_pred)
svm_precision = precision_score(y_test,y_pred)
svm_f1 = f1_score(y_test,y_pred)

model_result_accuracy.append(svm_accuracy)
model_result_f1.append(svm_f1)
model_result_precision.append(svm_precision)
model_result_recall.append(svm_recall) 

print('Support Vector Machines')

print('ACCURACY : ', svm_accuracy)
print('F1       : ', svm_f1)
print('PRECISION: ', svm_precision)
print('RECALL   : ', svm_recall)

confusionMatrix (y_test, y_pred,"Support Vector Machines")
rocCurve(svm,"Support Vector Machines", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''

#Decision Tree
classifiers.append("DecisionTree")
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)

dt_accuracy = accuracy_score(y_test,y_pred)
dt_recall = recall_score(y_test, y_pred)
dt_precision = precision_score(y_test,y_pred)
dt_f1 = f1_score(y_test, y_pred)

model_result_accuracy.append(dt_accuracy)
model_result_f1.append(dt_f1)
model_result_precision.append(dt_precision)
model_result_recall.append(dt_recall)

print('Decision Tree')

print('ACCURACY : ', dt_accuracy)
print('F1       : ', dt_f1)
print('PRECISION: ', dt_precision)
print('RECALL   : ', dt_recall)

confusionMatrix (y_test, y_pred,"Decison Tree")
rocCurve(dt,"Decision Tree", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''

#RandomForest
classifiers.append("RandomForest")
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

rf_accuracy = accuracy_score(y_test,y_pred)
rf_recall = recall_score(y_test,y_pred)
rf_precision = precision_score(y_test,y_pred)
rf_f1 = f1_score(y_test,y_pred)

model_result_accuracy.append(rf_accuracy)
model_result_f1.append(rf_f1)
model_result_precision.append(rf_precision)
model_result_recall.append(rf_recall)

print('Random Forest')

print('ACCURACY : ' , rf_accuracy)
print('F1       : ', rf_f1)
print('PRECISION: ', rf_precision)
print('RECALL   : ', rf_recall)

confusionMatrix (y_test, y_pred,"Random Forest")
rocCurve(rf,"Random Forest", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''


data = pd.DataFrame({'Classifier': classifiers, 'Test Accuracy': model_result_accuracy})
metricsVisualization (data, "Accuracy")

data = pd.DataFrame({'Classifier': classifiers, 'Test F1': model_result_f1})
metricsVisualization (data, "F1")

data = pd.DataFrame({'Classifier': classifiers, 'Test Precision': model_result_precision})
metricsVisualization (data, "Precision")

data = pd.DataFrame({'Classifier': classifiers, 'Test Recall': model_result_recall})
metricsVisualization (data, "Recall")

'''--------------------------------------------------------------------------------------------------------------------------'''

#LogisticRegression Hyperparameter

param_grid = [
    {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty' : ['l2'],'max_iter' : [100,1000,5000]},
    {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty' : ['l1'],'max_iter' : [100,1000,5000]}
    ]
    
lg=LogisticRegression(solver='liblinear')

grid_search = GridSearchCV(lg,param_grid,cv=5)
grid_search.fit(X_train,y_train)

print("Best hyperparameters Logistic Regression: ", grid_search.best_params_)
best_lg_model = grid_search.best_estimator_
best_lg_model.fit(X_train,y_train)
y_pred = best_lg_model.predict(X_test)

best_lg_model_accuracy = accuracy_score(y_test,y_pred)
best_lg_model_f1 = f1_score(y_test,y_pred)
best_lg_model_precision = precision_score(y_test,y_pred)
best_lg_model_recall = recall_score(y_test,y_pred)

model_result_accuracy_hyperparameters.append(best_lg_model_accuracy)
model_result_f1_hyperparameters.append(best_lg_model_f1)
model_result_precision_hyperparameters.append(best_lg_model_precision)
model_result_recall_hyperparameters.append(best_lg_model_recall)

print('Logistic Regression Hyperparameter')

print('ACCURACY : ', best_lg_model_accuracy)
print('F1       : ', best_lg_model_f1)
print('PRECISION: ', best_lg_model_precision)
print('RECALL   : ', best_lg_model_recall)

confusionMatrix (y_test, y_pred,"Logistic Regression Hyperparameter")
rocCurve(best_lg_model,"Logistic Regression Hyperparameter", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''

#K-Nearest Neighbors Hyperparameter
k_values = [i for i in range (1,20) ]
scores = []

for k in k_values:
    knnHype = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knnHype, X_train, y_train, cv = 5)
    scores.append(np.mean(score))
    
plt.plot(k_values, scores, color='g')
plt.xticks(ticks=k_values, labels=k_values)
plt.title("Number of Neighbors")
plt.grid()
plt.show()

best_index = np.argmax(scores)
best_k = k_values[best_index]

param_grid = {
    'weights':['uniform','distance'],
    'algorithm':['auto','ball_tree','kd_tree','brute'],
    'leaf_size':[10,20,30,40,50],
    'p':[1,2],
    'metric': ['euclidean','manhattan','minkowski'],
}

knnHype = KNeighborsClassifier(n_neighbors=best_k)

grid_search = GridSearchCV(knnHype,param_grid,cv=5)
grid_search.fit(X_train,y_train)

print("Best hyperparameters K-Nearest Neighbors: ", grid_search.best_params_)
best_knn_model = grid_search.best_estimator_
best_knn_model.fit(X_train,y_train)
y_pred = best_knn_model.predict(X_test)

best_knn_model_accuracy = accuracy_score(y_test,y_pred)
best_knn_model_f1 = f1_score(y_test,y_pred)
best_knn_model_precision = precision_score(y_test,y_pred)
best_knn_model_recall = recall_score(y_test,y_pred)

model_result_accuracy_hyperparameters.append(best_knn_model_accuracy)
model_result_f1_hyperparameters.append(best_knn_model_f1)
model_result_precision_hyperparameters.append(best_knn_model_precision)
model_result_recall_hyperparameters.append(best_knn_model_recall)

print('K-Nearest Neighbors Hyperparameter')

print('ACCURACY : ', best_knn_model_accuracy)
print('F1       : ', best_knn_model_f1)
print('PRECISION: ', best_knn_model_precision)
print('RECALL   : ', best_knn_model_recall)

confusionMatrix (y_test, y_pred,"K-Nearest Neighbors Hyperparameter")
rocCurve(best_knn_model,"K-Nearest Neighbors Hyperparameter", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''

#Support Vector Machines Hyperparamaters
parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
               {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
               {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05]} 
              ]
svc=SVC(probability=True) 
grid_search = GridSearchCV(estimator = svc,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)
print("Best hyperparameters Support Vector Machines: ", grid_search.best_params_)

best_svm_model= grid_search.best_estimator_
best_svm_model.fit(X_train, y_train)
y_pred = best_svm_model.predict(X_test)

best_svm_model_accuracy = accuracy_score(y_test,y_pred)
best_svm_model_f1 = f1_score(y_test,y_pred)
best_svm_model_precision = precision_score(y_test,y_pred)
best_svm_model_recall = recall_score(y_test,y_pred)

model_result_accuracy_hyperparameters.append(best_svm_model_accuracy)
model_result_f1_hyperparameters.append(best_svm_model_f1)
model_result_precision_hyperparameters.append(best_svm_model_precision)
model_result_recall_hyperparameters.append(best_svm_model_recall)

print('Support Vector Machines Hyperparameter')

print('ACCURACY : ', best_svm_model_accuracy)
print('F1       : ', best_svm_model_f1)
print('PRECISION: ', best_svm_model_precision)
print('RECALL   : ', best_svm_model_recall)

confusionMatrix (y_test, y_pred,"Support Vector Machines Hyperparameter")
rocCurve(best_svm_model,"Support Vector Machines Hyperparameter", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''

#DecisionTree
param_grid = {
    'criterion' : ['gini', 'entropy','log_loss'],
    'splitter': ['best','random'],
    'max_depth': [10, 20, 30, 40, 50],       
    'min_samples_split': [2, 5, 10, 20],     
    'min_samples_leaf': [1, 2, 4, 7]
}

dtHype=DecisionTreeClassifier()

grid_search = GridSearchCV(dtHype,param_grid,cv=5)
grid_search.fit(X_train,y_train)

print("Best hyperparameters Decision Tree: ", grid_search.best_params_)
best_dt_model = grid_search.best_estimator_
best_dt_model.fit(X_train,y_train)
y_pred = best_dt_model.predict(X_test)

best_dt_model_accuracy = accuracy_score(y_test,y_pred)
best_dt_model_f1 = f1_score(y_test,y_pred)
best_dt_model_precision = precision_score(y_test,y_pred)
best_dt_model_recall = recall_score(y_test,y_pred)

model_result_accuracy_hyperparameters.append(best_dt_model_accuracy)
model_result_f1_hyperparameters.append(best_dt_model_f1)
model_result_precision_hyperparameters.append(best_dt_model_precision)
model_result_recall_hyperparameters.append(best_dt_model_recall)

print('Decision Tree Hyperparameter')

print('ACCURACY : ', best_dt_model_accuracy)
print('F1       : ' , best_dt_model_f1)
print('PRECISION: ' , best_dt_model_precision)
print('RECALL   : ' , best_dt_model_recall)

confusionMatrix (y_test, y_pred,"Decision Tree Hyperparameter")
rocCurve(best_dt_model,"Decision Tree Hyperparameter", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''

#RandomForest
param_grid = { 'criterion' : ['gini', 'entropy','log_loss'],
              'n_estimators': [25, 50, 75, 100, 150, 200, 250] }

rfHype = RandomForestClassifier()

grid_search = GridSearchCV(rfHype,param_grid,cv=5)
grid_search.fit(X_train,y_train)

print("Best hyperparameters Random Forest: ", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train,y_train)
y_pred = best_rf_model.predict(X_test)

best_rf_model_accuracy = accuracy_score(y_test,y_pred)
best_rf_model_f1 = f1_score(y_test,y_pred)
best_rf_model_precision = precision_score(y_test,y_pred)
best_rf_model_recall = recall_score(y_test,y_pred)

model_result_accuracy_hyperparameters.append(best_rf_model_accuracy)
model_result_f1_hyperparameters.append(best_rf_model_f1)
model_result_precision_hyperparameters.append(best_rf_model_precision)
model_result_recall_hyperparameters.append(best_rf_model_recall)

print('Random Forest Hyperparameter')

print('ACCURACY : ' , best_rf_model_accuracy)
print('F1       : ' , best_rf_model_f1)
print('PRECISION: ' , best_rf_model_precision)
print('RECALL   : ' , best_rf_model_recall)

confusionMatrix (y_test, y_pred,"Random Forest Hyperparameter")
rocCurve(best_rf_model,"Random Forest Hyperparameter", y_test, X_test)

'''--------------------------------------------------------------------------------------------------------------------------'''

data = pd.DataFrame({'Classifier': classifiers, 'Test Accuracy': model_result_accuracy_hyperparameters})
metricsVisualizationHyperparameters (data, "Accuracy")

data = pd.DataFrame({'Classifier': classifiers, 'Test F1': model_result_f1_hyperparameters})
metricsVisualizationHyperparameters (data, "F1")

data = pd.DataFrame({'Classifier': classifiers, 'Test Precision': model_result_precision_hyperparameters})
metricsVisualizationHyperparameters (data, "Precision")

data = pd.DataFrame({'Classifier': classifiers, 'Test Recall': model_result_recall_hyperparameters})
metricsVisualizationHyperparameters (data, "Recall")

'''--------------------------------------------------------------------------------------------------------------------------'''
rocCurveComparison(log_reg, best_lg_model, X_test, y_test,'Logistic Regression Hyperparameters ', 'Logistic Regression ')
rocCurveComparison(knn, best_knn_model, X_test, y_test,'K-Nearest Neighbors Hyperparameters ', 'K-Nearest Neighbors ')
rocCurveComparison(svm, best_svm_model, X_test, y_test,'Support Vector Machines Hyperparameters ', 'Support Vector Machines ')
rocCurveComparison(dt, best_dt_model, X_test, y_test,'Decision Tree Hyperparameters ', 'Decision Tree ')
rocCurveComparison(rf, best_rf_model, X_test, y_test,'Random Forest Hyperparameters ', 'Random Forest ')

'''--------------------------------------------------------------------------------------------------------------------------'''

hypeVsNoHypeVisualization(classifiers, model_result_accuracy, model_result_accuracy_hyperparameters, "Accuracy")
hypeVsNoHypeVisualization(classifiers, model_result_f1, model_result_f1_hyperparameters, "F1")
hypeVsNoHypeVisualization(classifiers, model_result_precision, model_result_precision_hyperparameters, "Precision")
hypeVsNoHypeVisualization(classifiers, model_result_recall, model_result_recall_hyperparameters, "Recall")

'''--------------------------------------------------------------------------------------------------------------------------'''


df_RBayes = pd.DataFrame(np.array(dataset.copy(), dtype=int), columns=dataset.columns)

k2 = K2Score(df_RBayes)
hc_k2 = HillClimbSearch(df_RBayes)
modello_k2 = hc_k2.estimate(scoring_method=k2)

print(modello_k2.nodes())  

print(modello_k2.edges())

rete_bayesiana = BayesianNetwork(modello_k2.edges())
rete_bayesiana.fit(df_RBayes)

inferenza = VariableElimination(rete_bayesiana)
prob_nothd = inferenza.query(variables=['output'],
                              evidence={'sex': 1, 'cp': 2, 'chol': 255,
                                         'restecg': 1, 'thalachh': 175, 'exng': 0,
                                        'oldpeak': 0, 'slp': 2, 'caa': 2, 'thall': 2})

print('\nProbabilità per un individuo di non avere un problema cardiaco: ')
print(prob_nothd, '\n')

prob_hd = inferenza.query(variables=['output'],
                            evidence={'sex': 1 , 'cp': 0, 'chol': 274,
                                        'restecg': 0, 'thalachh': 166, 'exng': 0,
                                        'oldpeak': 0, 'slp': 1, 'caa': 0, 'thall': 3})

print('\nProbabilità per un individuo di avere un problema cardiaco: ')
print(prob_hd, '\n\n')

