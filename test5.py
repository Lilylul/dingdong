import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
# print(iris)
# print(iris.keys())
# print(iris.feature_names)
# print(iris.target)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,
                                                    test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
for i in (1,3,5,7): #k의 갯수
    for j in ('uniform', 'distance'): #거리계산
        for k in ('auto','ball_tree','kd_tree','brute'): #algorithm
            model = KNeighborsClassifier(n_neighbors=i,weights=j,algorithm=k)
            model.fit(X_train,y_train)
            y_p = model.predict(X_test)
            relation_square = model.score(X_test,y_test)
            from sklearn.metrics import confusion_matrix, classification_report
            knn_matrix = confusion_matrix(y_test,y_p)
            print(knn_matrix)
            taget_names = ['setosa','versicolor', 'virginica']
            knn_result = classification_report(y_test,y_p,target_names=taget_names)
            print(knn_result)
            print('accuracy : {:2f}'.format(model.score(X_test,y_test)))
        print('\n')
    print('\n')