import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import time
from datetime import datetime
import warnings

from pandas._libs.tslibs.timestamps import Timestamp
import jieba
import jieba.posseg

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier)
from sklearn import preprocessing
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

with open('train.pkl', 'rb') as file:
    train = pickle.load(file)

type_var = ['航空公司', '时间段', '机场', '航班编号', '天气']
num_var = ['飞机编号', '特情紧急程度', '最高气温', '最低气温', '特情内容数量', '计划飞行时间', '月份']

train = train.sort_values(by='时间')

train.drop('时间', axis=1, inplace=True)
train = train.dropna()

for var in type_var:
    le = preprocessing.LabelEncoder()
    le.fit(train[var])
    train[var] = le.transform(train[var])

train_ratio = 0.8
train_ = train.copy(deep=True)
train = train_.iloc[:int(train_.shape[0] * train_ratio), :]
test = train_.iloc[int(train_.shape[0] * train_ratio):, :]

train_x = train.drop('y', axis=1)
train_y = train['y']
test_x = test.drop('y', axis=1)
test_y = test['y']

def val(model):
    model.fit(train_x, train_y)
    predict_value = model.predict_proba(test_x)[:, 1]
    predict_bool = predict_value > 0.5
    true_value = test_y
    auc = roc_auc_score(true_value, predict_value)
    acc = precision_score(true_value, predict_bool)
    recall = recall_score(true_value, predict_bool)
    f1 = f1_score(true_value, predict_bool)
    return (auc, acc, recall, f1, predict_value)

test_auc_dict = {}
test_acc_dict = {}
test_recall_dict = {}
test_f1_dict = {}
predict_value_dict = {}

rf = RandomForestClassifier()
rf_auc, rf_acc, rf_recall, rf_f1, rf_predict_value = val(rf)

test_auc_dict['Random Forest'] = rf_auc
test_acc_dict['Random Forest'] = rf_acc
test_recall_dict['Random Forest'] = rf_recall
test_f1_dict['Random Forest'] = rf_f1
predict_value_dict['Random Forest'] = rf_predict_value
print('The auc score of rf is ' + str(round(rf_auc.mean(), 4)))

gbc = GradientBoostingClassifier()
gbc_auc, gbc_acc, gbc_recall, gbc_f1, gbc_predict_value = val(gbc)

test_auc_dict['Gradient Boosting'] = gbc_auc
test_acc_dict['Gradient Boosting'] = gbc_acc
test_recall_dict['Gradient Boosting'] = gbc_recall
test_f1_dict['Gradient Boosting'] = gbc_f1
predict_value_dict['Gradient Boosting'] = gbc_predict_value
print('The auc score of gbc is ' + str(round(gbc_auc.mean(), 4)))

ada = AdaBoostClassifier()
ada_auc, ada_acc, ada_recall, ada_f1, ada_predict_value = val(ada)

test_auc_dict['AdaBoost'] = ada_auc
test_acc_dict['AdaBoost'] = ada_acc
test_recall_dict['AdaBoost'] = ada_recall
test_f1_dict['AdaBoost'] = ada_f1
predict_value_dict['AdaBoost'] = ada_predict_value
print('The auc score of ada is ' + str(round(ada_auc.mean(), 4)))

xgbc = xgb.XGBClassifier()
xgbc_auc, xgbc_acc, xgbc_recall, xgbc_f1, xgbc_predict_value = val(xgbc)

test_auc_dict['Xgboost'] = xgbc_auc
test_acc_dict['Xgboost'] = xgbc_acc
test_recall_dict['Xgboost'] = xgbc_recall
test_f1_dict['Xgboost'] = xgbc_f1
predict_value_dict['Xgboost'] = xgbc_predict_value
print('The auc score of xgb is ' + str(round(xgbc_auc.mean(), 4)))

lgbc = lgb.LGBMClassifier()
lgbc_auc, lgbc_acc, lgbc_recall, lgbc_f1, lgbc_predict_value  = val(lgbc)

test_auc_dict['LightGBM'] = lgbc_auc
test_acc_dict['LightGBM'] = lgbc_acc
test_recall_dict['LightGBM'] = lgbc_recall
test_f1_dict['LightGBM'] = lgbc_f1
predict_value_dict['LightGBM'] = lgbc_predict_value
print('The auc score of lgb is ' + str(round(lgbc_auc.mean(), 4)))

test_auc_df = pd.DataFrame(test_auc_dict, index = ['auc score'])
test_acc_df = pd.DataFrame(test_acc_dict, index = ['accuracy rate'])
test_recall_df = pd.DataFrame(test_recall_dict, index = ['recall rate'])
test_f1_df = pd.DataFrame(test_f1_dict, index = ['f1 score'])

result_df = pd.concat([test_auc_df, test_acc_df, test_recall_df, test_f1_df], axis = 0)
result_df.to_csv('model_compare.csv', index=True)

with open('predict_value.pkl', 'wb') as file:
    pickle.dump(predict_value_dict, file)
    
with open('true_value.pkl', 'wb') as file:
    pickle.dump(test_y.tolist(), file)

