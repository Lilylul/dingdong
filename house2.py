from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
housing = housing.frame

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def iqr_outliers():
    global housing
    outlier_list = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup'] #이상치

    for i, item in enumerate(outlier_list):
        Q1,Q3 = np.percentile(housing[item],[25,75]) #Q1: 하위 25%, Q3: 상위 25%
        IQR = Q3 - Q1
        lower_bound = Q1 -1.5*IQR #IQR 하한선
        upper_bound = Q3 +1.5*IQR #IQR 상한선
        lower_data = housing[item]>=lower_bound
        upper_data = housing[item]<=upper_bound
        outliers = len(housing [(housing[item]<lower_bound) | (housing[item]>upper_bound)])
        print(f'{item}:{outliers}')

        housing =housing[lower_data&upper_data]

housing_before = housing.copy()
iqr_outliers()
housing_after = housing.copy()
import matplotlib.pyplot as plt

from scipy import stats
def z_score_outliers():
    global housing
    features = housing.drop(columns = ['MedHouseVal','Longitude','Latitude'])
    z_scores = np.abs(stats.zscore(features))
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)
    print('Z-Score Outliers Count : ', outliers.sum())
    housing = housing.loc[~outliers]
z_score_outliers()

from sklearn.model_selection import train_test_split
x = housing.drop(columns=['MedHouseVal']) #입력데이터(특성) - medhouseval을 제외한 모든 특성
y = housing['MedHouseVal']

X_train, X_test, y_train, y_test =train_test_split(x,y,test_size= 0.2,random_state=42)
data_mse = {}
data_score = {}
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def linear_regreession_model():
    model = LinearRegression()
    #모듈을 호출>linear regression
    model.fit(X_train,y_train) #모의고사를 학습을 진행함. linear regression으로
    y_pred = model.predict(X_test)  #수능으로 테스트를 진행하는것

    mse = mean_squared_error(y_test,y_pred) #(실제값-예측값)^2>어제는 root를 해서 rmse를 만듬
    data_mse['linear_regression'] =mse
    #data_mse가 dictionary이기 때문에 linear_regression라는 feature 에 mse를 집어넣습니다.
    data_score['linear_regression'] = r2_score(y_test,y_pred)
    #data_score가 dictionary이기 때문에 linear_regression라는 feature 에 mse를 집어넣습니다.
    return y_pred 
    #y_pred값을 return 해주게됨. > 밖에서 값을 호출하기 위해 가져오는것입니다.


from sklearn.linear_model import Lasso
def lasso_regreession(): #feature selection>보장
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X_train,y_train)
    y_pred = lasso_reg.predict(X_test) #수능으로 테스트를 진행 하는것이고.
    
    mse = mean_squared_error(y_test,y_pred) #(실제값-예측값)^2>어제는 root를 해서 rmse를 만듬
    data_mse['lasso_regression'] =mse
    #data_mse가 dictionary이기 때문에 linear_regression라는 feature 에 mse를 집어넣습니다.
    data_score['lasso_regression'] = r2_score(y_test,y_pred)
    #data_score가 dictionary이기 때문에 linear_regression라는 feature 에 mse를 집어넣습니다.
    return y_pred 
    #y_pred값을 return 해주게됨. > 밖에서 값을 호출하기 위해 가져오는것입니다.


from sklearn.linear_model import Ridge
def ridge_regreession(): #feature selection>보장
    ridge_reg = Ridge(alpha=0.1)
    ridge_reg.fit(X_train,y_train)
    y_pred = ridge_reg.predict(X_test) #수능으로 테스트를 진행 하는것이고.
    
    mse = mean_squared_error(y_test,y_pred) #(실제값-예측값)^2>어제는 root를 해서 rmse를 만듬
    data_mse['ridge_regression'] =mse
    #data_mse가 dictionary이기 때문에 linear_regression라는 feature 에 mse를 집어넣습니다.
    data_score['ridge_regression'] = r2_score(y_test,y_pred)
    #data_score가 dictionary이기 때문에 linear_regression라는 feature 에 mse를 집어넣습니다.
    return y_pred 
    #y_pred값을 return 해주게됨. > 밖에서 값을 호출하기 위해 가져오는것입니다.

import pandas as pd
y_pred =linear_regreession_model() #함수를 호출
y_pred_lasso =lasso_regreession() #함수를 호출
y_pred_ridge =ridge_regreession() #함수를 호출
res_data = pd.DataFrame({'Actural':y_test, 'Predicted_regression': y_pred, 'Predicted_lasso': y_pred_lasso, 'Predict_ridge': y_pred_ridge})
#dataframe을 만드는 이유: 비교하기 위해서
# res_data['Error'] = res_data['Actural'] - res_data['Predicted']
#error를 실제값에서 - 예측값
print(res_data.head(5))
print("MSE: ", data_mse)
print('R^2 score', data_score)
