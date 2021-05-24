# # Data Import
import pandas as pd
import numpy as np
from sklearn import metrics
pd.options.mode.chained_assignment = None

main_data = pd.read_csv("C:/Users/Asus/CSE475 project/diabetes.csv")
main_data.head()



# # Data Preprocessing
import statistics

label_array = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
for i in range(1, 6):
    label = label_array[i]
    x = []
    
    for j in range(0, len(main_data)):
        if(main_data[label][j] != 0):
            x.append(main_data[label][j])
        
    med = statistics.median(x)
    
    for j in range(0, len(main_data)):
        if(main_data[label][j] == 0):
            main_data[label][j] = med
    
main_data.head()

x = main_data.iloc[:, :-1]
y = main_data.iloc[:, 8:]



# # Splitting Train and Test Data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
y_test = y_test.to_numpy().ravel()
y_train = y_train.to_numpy().ravel()



# # Decision Tree
from sklearn import tree

decisionTreeClf = tree.DecisionTreeClassifier(random_state=0)
decisionTreeClf.fit(x_train, y_train)
decisionTreeprediction = decisionTreeClf.predict(x_test)
r1 = metrics.classification_report(y_test, decisionTreeprediction)
print(r1)

decision_tree_confusion_metrix = metrics.confusion_matrix(y_test, decisionTreeprediction)
pd.DataFrame(decision_tree_confusion_metrix)



# # Random Forest
from sklearn.ensemble import RandomForestClassifier

randomForestClf = RandomForestClassifier(random_state=0)
randomForestClf.fit(x_train, y_train)
randomForestPrediction = randomForestClf.predict(x_test)
r2 = metrics.classification_report(y_test, randomForestPrediction)
print(r2)

random_forest_confusion_metrix = metrics.confusion_matrix(y_test, randomForestPrediction)
pd.DataFrame(random_forest_confusion_metrix)



# # K Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier

knnClf = KNeighborsClassifier(n_neighbors=7)
knnClf.fit(x_train, y_train)
knnPrediction = knnClf.predict(x_test)
r3 = metrics.classification_report(y_test, knnPrediction)
print(r3)

knn_confusion_metrix = metrics.confusion_matrix(y_test, knnPrediction)
pd.DataFrame(knn_confusion_metrix)



# # Support vector machine
from sklearn import svm

svmClf = svm.SVC()
svmClf.fit(x_train, y_train)
svmPrediction = svmClf.predict(x_test)
f4 = metrics.classification_report(y_test, svmPrediction)
print(f4)

svm_confusion_metrix = metrics.confusion_matrix(y_test, svmPrediction)
pd.DataFrame(svm_confusion_metrix)