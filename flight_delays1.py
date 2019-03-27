import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
print("Read csv files...")
train_df = pd.read_csv('flight\\flight_delays_train.csv')
test_df = pd.read_csv('flight\\flight_delays_test.csv')
X_train = train_df[['Distance', 'DepTime']].values 
y_train = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
X_test = test_df[['Distance', 'DepTime']].values

X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=17)
logit_pipe = Pipeline([('scaler', StandardScaler()), ('logit', LogisticRegression(C=1, random_state=17, solver='liblinear'))])
logit_pipe.fit(X_train_part, y_train_part)
logit_valid_pred = logit_pipe.predict_proba(X_valid)[:, 1]
print("ROC AUC BASE:"+str(roc_auc_score(y_valid, logit_valid_pred)))
#Route
train_df['Route'] = train_df['Origin'] + "-" +train_df['Dest']
test_df['Route'] = test_df['Origin'] + "-" +test_df['Dest']

ohe = OneHotEncoder(sparse = False)
print("Add month features...")
month_ohe_feat_train = ohe.fit_transform(train_df['Month'].values.reshape(-1,1))
month_df_train = pd.DataFrame(month_ohe_feat_train,columns=["month="+str(i) for i in range(month_ohe_feat_train.shape[1])])
train_df = pd.concat([train_df, month_df_train], axis=1)
month_ohe_feat_test = ohe.fit_transform(test_df['Month'].values.reshape(-1,1))
month_df_test = pd.DataFrame(month_ohe_feat_test,columns=["month="+str(i) for i in range(month_ohe_feat_test.shape[1])])
test_df = pd.concat([test_df, month_df_test], axis=1)
print("Add DayOfWeek features...")
day_ohe_feat_train = ohe.fit_transform(train_df['DayOfWeek'].values.reshape(-1,1))
day_df_train = pd.DataFrame(day_ohe_feat_train,columns=["day="+str(i) for i in range(day_ohe_feat_train.shape[1])])
train_df = pd.concat([train_df, day_df_train], axis=1)
day_ohe_feat_test = ohe.fit_transform(test_df['DayOfWeek'].values.reshape(-1,1))
day_df_test = pd.DataFrame(day_ohe_feat_test,columns=["day="+str(i) for i in range(day_ohe_feat_test.shape[1])])
test_df = pd.concat([test_df, day_df_test], axis=1)

ohe = OneHotEncoder(sparse = True)
od_ohe_feat_train = ohe.fit_transform(train_df['DayOfWeek'].values.reshape(-1,1))


print(train_df.columns.values.tolist())
columns = test_df.columns.values.tolist()
columns.remove('Month')
columns.remove('DayofMonth')
columns.remove('DayOfWeek')
columns.remove('UniqueCarrier')
columns.remove('Origin')
columns.remove('Dest')
columns.remove('OriginDest')
print("Split train/test parts...")
X_train = train_df[columns].values
X_test = test_df[columns].values
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=17)
print("Fit and predict...")
logit_pipe = Pipeline([('scaler', StandardScaler()), ('logit', LogisticRegression(C=1, random_state=17, solver='liblinear'))])
logit_pipe.fit(X_train_part, y_train_part)
logit_valid_pred = logit_pipe.predict_proba(X_valid)[:, 1]
print("ROC AUC NEW:"+str(roc_auc_score(y_valid, logit_valid_pred)))