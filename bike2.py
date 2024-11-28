import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

df = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

#print(df.describe()) #결측치 없음
# #---------------------------------------------------------------------------------
#              season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
# count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
# mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
# std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
# min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
# 25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
# 50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
# 75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
# max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000

#print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10886 entries, 0 to 10885
# Data columns (total 12 columns):
# #   Column      Non-Null Count  Dtype
# # ---  ------      --------------  -----
#  0   datetime    10886 non-null  object
#  1   season      10886 non-null  int64
#  2   holiday     10886 non-null  int64
#  3   workingday  10886 non-null  int64
#  4   weather     10886 non-null  int64
#  5   temp        10886 non-null  float64
#  6   atemp       10886 non-null  float64
#  7   humidity    10886 non-null  int64
#  8   windspeed   10886 non-null  float64
#  9   casual      10886 non-null  int64
#  10  registered  10886 non-null  int64
#  11  count       10886 non-null  int64
# dtypes: float64(3), int64(8), object(1)
# memory usage: 1020.7+ KB
#print(info())

#print(df['datetime'])
# #-------------------------------------
# 0        2011-01-01 00:00:00
# 1        2011-01-01 01:00:00
# 2        2011-01-01 02:00:00
# 3        2011-01-01 03:00:00
# 4        2011-01-01 04:00:00
#                 ...
# 10881    2012-12-19 19:00:00
# 10882    2012-12-19 20:00:00
# 10883    2012-12-19 21:00:00
# 10884    2012-12-19 22:00:00
# 10885    2012-12-19 23:00:00
# Name: datetime, Length: 10886, dtype: object
# import calendar

# df['date'] = df.datetime.apply(lambda x: x.split()[0])
# df['hour'] = df.datetime.apply(lambda x: x.split()[1].split(".")[0])
# df['weekday'] = df.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
# df['month'] = df.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
# df["season"] = df.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })
# df["weather"] = df.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\
#                                         2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \
#                                         3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \
#                                         4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })

# #카테고리 유형으로 강제 변환
# categoryVariableList = ["hour", 'weekday', 'month', 'season', 'weather', 'holiday', 'workingday']
# for var in categoryVariableList:
#     df[var] = df[var].astype('category')

# import missingno as msno # 결측치를 보는 plot
# msno.matrix(df,figsize=(12,5))
# import matplotlib.pyplot as plt
# plt.show()

#연도만 출력
#df['datetime'] =pd.to_datetime(df['datetime'])
#df['year'] = df['datetime'].dt.year
#df['month'] = df['datetime'].dt.month
#df['day'] = df['datetime'].dt.day
#df['hour'] = df['datetime'].dt.hour
#df['minute'] = df['datetime'].dt.minute
#df['second'] = df['datetime'].dt.second
#요일 데이터 = -일요일은 6 (0은 월요일)
#df['dayofweek'] = df['datetime'].dt.dayofweek

# figure, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3) #subplot은 미리 지정해야함
# figure.set_size_inches(18,8) #인치

# sns.barplot(data = df, x= 'year', y='count', ax= ax1)
# sns.barplot(data = df, x= 'month', y='count', ax= ax2)
# sns.barplot(data = df, x= 'day', y='count', ax= ax3)
# sns.barplot(data = df, x= 'hour', y='count', ax= ax4)
# sns.barplot(data = df, x= 'minute', y='count', ax= ax5)
# sns.barplot(data = df, x= 'second', y='count', ax= ax6)

# ax1.set(ylabel='Count', title='Year rental amount')
# ax2.set(ylabel='month', title='Month rental amount')
# ax3.set(ylabel='day', title='Day rental amount')
# ax4.set(ylabel='hour', title='Hour rental amount')

# fig, axes = plt.subplot(nrows=2, ncols=2)
# fig.set_size_inches(12,10)
# sns.boxplot(data = df, y = 'count', orient= 'v', ax = axes[0][0])
# sns.boxplot(data = df, y = 'count', orient= 'v', x ='season' ax = axes[0][1])
# sns.boxplot(data = df, y = 'count', orient= 'v', x ='hour' ax = axes[1][0])
# sns.boxplot(data = df, y = 'count', orient= 'v', x = 'workingday' , ax = axes[1][2])

# axes[0][0].set(ylabel='Count', title = 'Rental amount')
# axes[0][1].set(ylabel='Season',ylabel='Count', title = 'Seasonal Rental amount')
# axes[1][0].set(ylabel='Hour of The Day',ylabel='Count', title = 'Hour Rental amount')
# axes[1][1].set(ylabel='Working Day',ylabel='Count', title = 'Working or not Rental amount')
# #이상치와 데이터의 균등성을 유추 가능

# plt.show()

# fig,(ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows = 5)
# fig.set_size_inches(18, 25)

# #꺽은 선 그래프
# sns.pointplot(data = df, x= 'hour', y='count', ax=ax1)
# sns.pointplot(data = df, x= 'hour', y='count', hue= 'workingday', ax=ax2)
# sns.pointplot(data = df, x= 'hour', y='count', hue= 'dayofweek', ax=ax3)
# sns.pointplot(data = df, x= 'hour', y='count', hue= 'weather', ax=ax4)
# sns.pointplot(data = df, x= 'hour', y='count', hue= 'season', ax=ax5)

# plt.show()

# corrMatt = df.corr()
# mask = np.array(corrMatt)
# #Return the indices for upper-triangle of arr.
# # 상삼각 행렬 False -> 하삼각 행렬 = Truie
# mask[np.tril_indices_from(mask)] = False

# fig, ax = plt.subplots()
# fig.set_size_inches(20,10)
# sns.heatmap(corrMatt, mask = mask, vmax = 0.8, square = True, annot = True)


# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
# fig.set_size_inches(12,5)
# sns.regplot(x='temp', y ='count', data=df, ax=ax1)
# sns.regplot(x='windspeed', y ='count', data=df, ax=ax2)
# sns.regplot(x='humidity', y ='count', data=df, ax=ax3)
# plt.show()

# def concatenate_year_month(datetime):
#     return "{0}-{1}".format(datetime.year, datetime.month)
# df['year_month'] = df['datetime'].apply(concatenate_year_month)
# print(df[['datetime', 'year_month']])

# fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2)
# fig.set_size_inches(18,4)

# sns.barplot(data =df, x='year', y='count', ax= ax1)
# sns.barplot(data =df, x='month', y='count', ax= ax2)

# fig, ax3 = plt.subplots(nrows=1, ncols=1)
# fig.set_size_inches(18,4)
# sns.barplot(data=df, x='year_month', y='count', ax= ax3 )
#plt.show()

##이상치 처리 (IQR, 3-sigma)
#IQR = Q3(75%) -Q1(25%)
#Q1-1.5*IQR < x < Q3+ 1.5*IQR
#Q1−1.5∗IQR :   최소 제한선
#Q3+1.5∗IQR :   최대 제한선

# count_q1 = np.percentile(df['count'],25)
# count_q3 = np.percentile(df['count'],75)
# count_IQR = count_q3-count_q1
# # 이상치를 제외한(이상치가 아닌 구간에 있는) 데이터만 조회
# df_IQR = df[(df['count']>= (count_q1 - (1.5*count_IQR))) &
#             (df['count']<= (count_q3 + (1.5*count_IQR)))]
# #print(df_IQR) #300개 정도는 잃어버림

# #3-sigma (평균 +/- 표준편차 차이)
# df_sigma = df[np.abs(df['count']-df['count'].mean() <= 3*df['count'].std())]
# #print(df_sigma) #약 100개 정도 잃어 버림
 
# #IQR을 적용하여 그림 
# fig, axes = plt.subplots(nrows=2, ncols=2)
# fig.set_size_inches(12,10)
# sns.boxplot(data=df_IQR, y="count", orient= "v", ax=axes[0][0])
# sns.boxplot(data=df_IQR, y="count", x = "season",orient= "v", ax=axes[0][1])
# sns.boxplot(data=df_IQR, y="count", x="hour",orient= "v", ax=axes[1][0])
# sns.boxplot(data=df_IQR, y="count", x="workingday",orient= "v", ax=axes[1][1])

# axes[0][0].set(ylabel='Count',title="Rental amount")
# axes[0][1].set(xlabel='Season',ylabel='Count',title="Seasonal Rental amount")
# axes[1][0].set(xlabel='Hour of The Day',ylabel='Count',title="Hour Rental amount")
# axes[1][1].set(xlabel='Working Day',ylabel='Count',title="Working or not Rental amount")

#plt.show()

data = pd.concat([df,df_test])
data.reset_index(inplace=True)
data.drop('index', inplace = True, axis=1 )

data['date'] = data.datetime.apply(lambda x: x.split()[0])
data['hour'] = data.datetime.apply(lambda x: x.split()[1].split(":")[0]).astype('int')
data['year'] = data.datetime.apply(lambda x: x.split()[0].split("-")[0])
data['weekday'] = data.date.apply(lambda dateString : datetime.strptime(dateString, "%Y-%m-%d").weekday())
data['month'] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]
numericalFeatureNames = ["temp","humidity","windspeed","atemp"]
dropFeatures = ['casual',"count","datetime","date","registered"]

for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")

dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])
dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])
datetimecol = dataTest["datetime"]
yLabels = dataTrain["count"]
yLablesRegistered = dataTrain["registered"]
yLablesCasual = dataTrain["casual"]

dataTrain  = dataTrain.drop(dropFeatures,axis=1)
dataTest  = dataTest.drop(dropFeatures,axis=1)

def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

#np.nan_to_num : Replace NaN with zero and infinity with large finite numbers (default behaviour)
#or with the numbers defined by the user using the nan, posinf and/or neginf keywords.

np.log(np.NaN)

from sklearn.metrics import mean_squared_error,mean_absolute_error
def rmsle(y,pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y-log_pred)**2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle
#sklearn의 mean_squared_error 이용해 RMSE계산
def rmse(y,pred):
    return np.sqrt(mean_squared_error(y,pred))


#MSE, RMSE, RMSLE 계산
def evaluate_rgre(y,pred):
    rmsle_val = rmsle(y,pred)
    rmse_val = rmse(y,pred)
    mae_val = mean_absolute_error(y,pred)
    print('RMSLE:{0:.3f}, RMSE:{1:.3f}, MAE:{2:.3f}'.format(rmsle_val,rmse_val,mae_val))


#분리를 통해 추출된 속성은 문자열 속성을 가지고 있음 따라서 숫자형 데이터로 변환해 줄 필요가 있음.
#pandas.to_numeric(): https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_numeric.html
dataTrain['year'] = pd.to_numeric(dataTrain.year,errors='coerce')
dataTrain['month'] = pd.to_numeric(dataTrain.month,errors='coerce')
dataTrain['hour'] = pd.to_numeric(dataTrain.hour,errors='coerce')
dataTrain['weekday'] = pd.to_numeric(dataTrain.hour,errors='coerce')

dataTrain['season'] = pd.to_numeric(dataTrain.year,errors='coerce')
dataTrain['holiday'] = pd.to_numeric(dataTrain.month,errors='coerce')
dataTrain['workingday'] = pd.to_numeric(dataTrain.hour,errors='coerce')
dataTrain['weather'] = pd.to_numeric(dataTrain.hour,errors='coerce')

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logistic regression model
lModel = LinearRegression()

# Train the model
yLabelsLog = np.log1p(yLabels)
lModel.fit(X = dataTrain,y = yLabelsLog)

# Make predictions
preds = lModel.predict(X= dataTrain)
print ("RMSLE Value For Linear Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds)))


ridge_m_ = Ridge()
ridge_params_ = { 'max_iter':[3000],'alpha':[0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
grid_ridge_m = GridSearchCV( ridge_m_,
                          ridge_params_,
                          scoring = rmsle_scorer,
                          cv=5)
yLabelsLog = np.log1p(yLabels)
grid_ridge_m.fit( dataTrain, yLabelsLog )
preds = grid_ridge_m.predict(X= dataTrain)
print (grid_ridge_m.best_params_)
print ("RMSLE Value For Ridge Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds)))

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_ridge_m.cv_results_)
df["rmsle"] = df["mean_score_time"].apply(lambda x:-x)
sns.pointplot(data=df,x=df['param_alpha'],y="rmsle",ax=ax)


lasso_m_ = Lasso()

alpha  = 1/np.array([0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000])
lasso_params_ = { 'max_iter':[3000],'alpha':alpha}

grid_lasso_m = GridSearchCV( lasso_m_,lasso_params_,scoring = rmsle_scorer,cv=5)
yLabelsLog = np.log1p(yLabels)
grid_lasso_m.fit( dataTrain, yLabelsLog )
preds = grid_lasso_m.predict(X= dataTrain)
print (grid_lasso_m.best_params_)
print ("RMSLE Value For Lasso Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds)))

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_lasso_m.cv_results_)
df["rmsle"] = df["mean_score_time"].apply(lambda x:-x)
sns.pointplot(data=df,x=df['param_alpha'],y="rmsle",ax=ax)

plt.show()

