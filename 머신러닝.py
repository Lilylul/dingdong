#유방암 데이터
# from sklearn.tree import DecisionTreeClassifier
# #from sklearn.tree import DecisionTreeClassifier _회귀
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# cancer = load_breast_cancer()
# x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target,
#                                                     stratify=cancer.target, random_state=42)
# #stratify: target =>cancer.target이 있는데 왜 정합니까?
# #stratify: 정확한 target을 지정/ classification을 할때는 정해주는것이 좋음.
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(x_train,y_train)
# print('Accuracy on training set: {:.3f}'.format(clf.score(x_train,y_train)))
# print('Accuracy on test set: {:.3f}'.format(clf.score(x_test,y_test)))

# tree = DecisionTreeClassifier(max_depth=10)
# tree.fit(x_train,y_train)
# print('Accuracy on training set: {:.3f}'.format(tree.score(x_train,y_train)))
# print('Accuracy on test set: {:.3f}'.format(tree.score(x_test,y_test)))

# # from sklearn.tree import export_graphviz

# # export_graphviz(tree, out_file='tree.dot',class_names=['malinant','benign'],
# #                 feature_names=cancer.feature_names, impurity=False, filled=True)
# # import graphviz
# # with open('tree.dot') as f:
# #     dot_graph = f.read()
# # graphviz.Source(dot_graph).view()

# print('Feature importances:')
# print(tree.feature_importances_)

import numpy as np
import matplotlib.pyplot as plt
# def plot_feature_importances_cancer(model):
#      n_features = cancer.data.shape[1]
#      plt.barh(np.arange(n_features),model.feature_importances_,align='center')
#      plt.yticks(np.arange(n_features),cancer.feature_names)
#      plt.xlabel('Feature importance')
#      plt.ylabel('Feature')
#      plt.ylim(-1,n_features)

# plot_feature_importances_cancer(tree)
# plt.show()


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# import numpy as np
# import matplotlib.pyplot as plt
 
# cancer = load_breast_cancer()

# x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

# n_feature = cancer.data.shape[1]

# score_n_tr_est = [] 
# score_n_te_est = [] 
# score_m_tr_mft = []
# score_m_te_mft = []

# for i in np.arange(1, n_feature+1): # n_estimators와 max_features는 모두 0보다 큰 정수여야 하므로 1부터 시작합니다.

#     params_n = {'n_estimators':i, 'max_features':'sqrt', 'n_jobs':-1} # **kwargs parameter

#     params_m = {'n_estimators':10, 'max_features':i, 'n_jobs':-1}

#     forest_n = RandomForestClassifier(**params_n).fit(x_train, y_train) 

#     forest_m = RandomForestClassifier(**params_m).fit(x_train, y_train)


#     score_n_tr = forest_n.score(x_train, y_train)

#     score_n_te = forest_n.score(x_test, y_test)

#     score_m_tr = forest_m.score(x_train, y_train)

#     score_m_te = forest_m.score(x_test, y_test)



#     score_n_tr_est.append(score_n_tr)

#     score_n_te_est.append(score_n_te)

#     score_m_tr_mft.append(score_m_tr)

#     score_m_te_mft.append(score_m_te)


# index = np.arange(len(score_n_tr_est))

# plt.plot(index, score_n_tr_est, label='n_estimators train score', color='lightblue', ls='--') # ls: linestyle

# plt.plot(index, score_m_tr_mft, label='max_features train score', color='orange', ls='--')

# plt.plot(index, score_n_te_est, label='n_estimators test score', color='lightblue')

# plt.plot(index, score_m_te_mft, label='max_features test score', color='orange')

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),

#            ncol=2, fancybox=True, shadow=False) # fancybox: 박스모양, shadow: 그림자

# plt.xlabel('number of parameter', size=15)

# plt.ylabel('score', size=15)

# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
# print(type(cancer)) #bunch: 특이한 구조에 있음>sklearn의 고유한 특징

# print(dir(cancer))
# print(cancer.data.shape) #569x30///
# print(cancer.feature_names)
# print(cancer.target_names)
# print(cancer.target)
# print(np.bincount(cancer.target))
# print(cancer.DESCR)
for i, name in enumerate(cancer.feature_names):
    print('%02d: %s' %(i,name))
maliganant =cancer.data[cancer.target==0]
benign = cancer.data[cancer.target==1]

_, bin = np.histogram(cancer.data[:,0],bins=20)
print(np.histogram(cancer.data[:,0],bins=20))

# plt.hist(maliganant[:,0],bins=bin, alpha=0.3)
# plt.hist(benign[:,0],bins=bin,alpha=0.3)
# plt.title(cancer.feature_names[0])
# plt.show()

# plt.figure(figsize=(20,15))
# for col in range(30):
#     plt.subplot(8,4,col+1)
#     _,bins =np.histogram(cancer.data[:,col],bins=20)
#     plt.hist(maliganant[:,col], bins = bins, alpha=0.3)
#     plt.hist(benign[:,col], bins = bins, alpha=0.3)
#     plt.title(cancer.feature_names[col])
#     if col ==0:plt.legend(cancer.target_names)
#     plt.xticks([])
# plt.show()

# fig = plt.figure(figsize=(14,14))
# fig.suptitle('Breast Cancer - Feature analysis', fontsize=20)
# for col in range(cancer.feature_names.shape[0]):
#     plt.subplot(8,4,col+1)
#     plt.scatter(cancer.data[:,0], cancer.target, c=cancer.target, alpha=0.5)
#     plt.title(cancer.feature_names[col]+('(%d)' %col))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

from sklearn.linear_model import LogisticRegression #분류모델.

# scores = []

# for i in range(10):
#     x_train,x_test,y_train,y_test = train_test_split(cancer.data, cancer.target)
#     model = LogisticRegression()
#     model.fit(x_train,y_train)

#     score = model.score(x_test,y_test)
#     scores.append(score)
# print('scores =', scores)
import matplotlib.pyplot as plt
import numpy as np

fig,axes = plt.subplots(5,6,figsize = [12,20])
fig.suptitle('mean radius VS others', fontsize =20)

for i in range(30):
    ax = axes.ravel()[i] #flatten함수와 비슷한데, 다른게 있음> 학습을 해볼것 flattenj vs ravel 두개가 다름.
    ax.scatter(cancer.data[:,0], cancer.data[:,i],c=cancer.target, cmap='winter',alpha=0.1)
    ax.set_title(cancer.feature_names[i]+('\n(%d)' %i))
    ax.set_axis_off()
plt.show()
