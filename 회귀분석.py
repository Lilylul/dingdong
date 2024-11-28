import numpy as np
import pandas as pd
#Visualization Libraries
import seaborn as sns
#seaborn : 그래프를 통계적으로 그리는 패키지.
import matplotlib.pyplot as plt
# imoprt package
#->함수를 몽땅 가지고 오는 것,
#from package imoprt module
# 패키지안에서 특정함수를 딸랑만 가지고 오는것.

from sklearn import datasets
#imports from sklearn library
from sklearn.linear_model import LinearRegression
#from sklearn. ->.은 속성으로 들어가주세요!
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

X, y = datasets.fetch_openml('boston', return_X_y=True)

#print(X.describe())
#print(X.info())
#print(X.isnull().sum())
#(X.isna().sum())

# sns.set_theme(rc = {'figure.figsize':(15,10)})
# correlation_matrix = X.corr().round(2)
# sns.heatmap(data = correlation_matrix,annot = True)
#annot : 소수점 표기를 해볼 것이냐 ? annot = False
# plt.show()

# features = ['LSTAT','RM','DIS']

# for i, col in enumerate(features):
#     plt.subplot(1,len(features),i+1)
#     x = X[col]
#     #y = y
#     plt.scatter(x,x,marker = 'o', color = '#e35')
#     plt.title('Variation in House Price')
#     plt.xlabel(col)
#     plt.ylabel('House Price')
# plt.show()


# column_sels = ['LSTAT','INDUS','NOX','PTRATIO','RM','TAX','DIS','AGE']
# x = X.loc[:,column_sels] #loc : label location
# fig,axes = plt.subplots(ncols = 4, nrows = 2, figsize = (20,10))
# axs = axes.flatten()#1차원 벡터로 만들어주는 것 / np.ravel() 얕은/깊은 복사
# for i,k in enumerate(column_sels):
#     sns.regplot(y=y,x=x[k], ax = axs[i])
# plt.tight_layout(pad=0.4, w_pad=0.5,h_pad=0.5)#layout : 틀
# plt.show()


x = X.RM #(506,) -> 행 506/가변공간
#sklearn 가정 : 2D -> 1D 벡터 (n by 1 / 1 by n)
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

plt.figure(figsize=(20,5))

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size= 0.2 , random_state= 5)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


reg = LinearRegression()
reg.fit(X_train,y_train)
y_train_predict = reg.predict(np.array(X_test))
rmse = np.sqrt(mean_squared_error(y_train,y_train_predict))
r2 = round(reg.score(np.array(X_test),2))
print(r2)

prediction_space = np.linspace(min(X_train),max(X_train)).reshape(-1,1)
plt.scatter(X_train,y_train)
plt.plot(prediction_space, reg.predict(prediction_space),color='black', linewidth = 3)
plt.ylabel('value of house/1000')
plt.xlabel('number of rooms')
plt.show()