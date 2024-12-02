import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('./boston.csv')
x = df.drop(columns='MEDV')
y = df[['MEDV']]
#pip install statsmodels_ 설치

# import statsmodels.api as sm
# X_constant = sm.add_constant(x)
# model_1 = sm.OLS(y,X_constant)
# lin_reg=model_1.fit()
# print(lin_reg.summary())
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# lr = LinearRegression()
# neg_mse_scores = cross_val_score(lr,x,y,scoring='neg_mean_squared_error', cv=100)
# rmse_scores = np.sqrt(-1*neg_mse_scores)
# avg_rmse = np.mean(rmse_scores)
# print('5 folds의 개별 Negative MSE scores: ',np.round(neg_mse_scores))
# print('5 folds의 개별 RMSE scores: ',np.round(rmse_scores,2))
# print('5 folds의 평균 RMSE: {0:.3f}'.format(avg_rmse))


from sklearn.model_selection import KFold
import numpy as np

#예시 데이터
# X_data =np.arange(10).reshape(10,1)
# y_target = np.arange(10)

# #k_fold 객체 생성(5개의 객체로 나눔)
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# #Fold의 정보 확인
# fold_idx = 1
# for train_index, test_index in kf.split(X_data):
#     print(f'Fold{fold_idx}')
#     print(f'Train indices :  {train_index}')
#     print(f'Test indices : {test_index}')
#     print(f'Train data: {X_data[train_index].flatten()}')
#     print(f'Test data : {X_data[test_index].flatten()}')
#     fold_idx += 1

# from sklearn.preprocessing import PolynomialFeatures
# import numpy as np

# x = np.arange(4).reshape(2,2)
# print('일차 단항식 계수 feature: \n', x)

# #degree=2인 2차 다항식으로 변환하기 위해 Polynomial Features를 이용하여 변환.
# # poly =PolynomialFeatures(degree=2)
# poly =PolynomialFeatures(degree=3).fit_transform(x)
# # poly.fit(x)
# # poly_ftr = poly.transform(x) #fit만하고 변환 안하면 안됨
# #poly_ftr = poly.fit_transform #위의 두개를 하나로 합칠 수 있음.
# # print('변환된 2차 다항식 계수 feature: \n', poly_ftr) 
# print('변환된 3차 다항식 계수 feature: \n', poly)

# def polynomail_func(x):
#     y = 1 + 2*x + x**2 + x**3
#     return y
# y = polynomail_func(x)
# #linear regression에 3차 다항식 계수 feature와 3차 다항식 결정값으로 학습 후 회귀 계수 확인
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(poly,y)
# print('Polynomial 회귀 계수 : \n',np.round(model.coef_,2))
# print('Polynomial 회귀 shape : ', model.coef_.shape)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

def polynomial_func(x):
    y = 1+ 2*x + x**2 +x**3
    return y
#Pipeline 객체로 streamline 하게 polynomial Feature 변환과 Linear regression을 연결
model = Pipeline([('Poly',PolynomialFeatures(degree= 3)),
                  ('linear', LinearRegression())])
x = np.arange(4).reshape(2,2)
y = polynomial_func(x)

model = model.fit(x,y)
print('Polynomial 회귀 계수: \n', np.round(model.named_steps['linear'].coef_,2))