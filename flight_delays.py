import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
#Add day features
print("Add day features...")
train_df['Mon'] = train_df['DayOfWeek'].apply(lambda day: 1 if day=='c-1' else 0).astype('int')
train_df['Tue'] = train_df['DayOfWeek'].apply(lambda day: 1 if day=='c-2' else 0).astype('int')
train_df['Wed'] = train_df['DayOfWeek'].apply(lambda day: 1 if day=='c-3' else 0).astype('int')
train_df['Thu'] = train_df['DayOfWeek'].apply(lambda day: 1 if day=='c-4' else 0).astype('int')
train_df['Fri'] = train_df['DayOfWeek'].apply(lambda day: 1 if day=='c-5' else 0).astype('int')
train_df['Sat'] = train_df['DayOfWeek'].apply(lambda day: 1 if day=='c-6' else 0).astype('int')
train_df['Sun'] = train_df['DayOfWeek'].apply(lambda day: 1 if day=='c-7' else 0).astype('int')
#test
test_df['Mon'] = test_df['DayOfWeek'].apply(lambda day: 1 if day=='c-1' else 0).astype('int')
test_df['Tue'] = test_df['DayOfWeek'].apply(lambda day: 1 if day=='c-2' else 0).astype('int')
test_df['Wed'] = test_df['DayOfWeek'].apply(lambda day: 1 if day=='c-3' else 0).astype('int')
test_df['Thu'] = test_df['DayOfWeek'].apply(lambda day: 1 if day=='c-4' else 0).astype('int')
test_df['Fri'] = test_df['DayOfWeek'].apply(lambda day: 1 if day=='c-5' else 0).astype('int')
test_df['Sat'] = test_df['DayOfWeek'].apply(lambda day: 1 if day=='c-6' else 0).astype('int')
test_df['Sun'] = test_df['DayOfWeek'].apply(lambda day: 1 if day=='c-7' else 0).astype('int')
#Add month features
print("Add month features...")
train_df['Jan'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-1' else 0).astype('int')
train_df['Feb'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-2' else 0).astype('int')
train_df['Mar'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-3' else 0).astype('int')
train_df['Apr'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-4' else 0).astype('int')
train_df['May'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-5' else 0).astype('int')
train_df['Jun'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-6' else 0).astype('int')
train_df['Jul'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-7' else 0).astype('int')
train_df['Aug'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-8' else 0).astype('int')
train_df['Sep'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-9' else 0).astype('int')
train_df['Oct'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-10' else 0).astype('int')
train_df['Nov'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-11' else 0).astype('int')
train_df['Dec'] = train_df['Month'].apply(lambda mon: 1 if mon=='c-12' else 0).astype('int')
#test
test_df['Jan'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-1' else 0).astype('int')
test_df['Feb'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-2' else 0).astype('int')
test_df['Mar'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-3' else 0).astype('int')
test_df['Apr'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-4' else 0).astype('int')
test_df['May'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-5' else 0).astype('int')
test_df['Jun'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-6' else 0).astype('int')
test_df['Jul'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-7' else 0).astype('int')
test_df['Aug'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-8' else 0).astype('int')
test_df['Sep'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-9' else 0).astype('int')
test_df['Oct'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-10' else 0).astype('int')
test_df['Nov'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-11' else 0).astype('int')
test_df['Dec'] = test_df['Month'].apply(lambda mon: 1 if mon=='c-12' else 0).astype('int')

y_train = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
full_df = pd.concat([train_df.drop('dep_delayed_15min', axis=1), test_df])
unique_carriers = full_df['UniqueCarrier'].unique()
unique_dest = full_df['Dest'].unique()
unique_origin = full_df['Origin'].unique()
full_df.drop(full_df.index, inplace=True)
print("Number of unique carriers:"+str(len(unique_carriers)))
print("Number of unique dest airports:"+str(len(unique_dest)))
def code_ohe_carr(data, feature):
    for carr in unique_carriers:
        data[carr] = (data[feature]==carr).astype('float')
def code_ohe_dest(data, feature):
    for dest in unique_dest:
        data[dest] = (data[feature]==dest).astype('float')
def code_ohe_origin(data, feature):
    for origin in unique_origin:
        data["orig_"+origin] = (data[feature]==origin).astype('float')
print("Add UniqueCarrier features...")
code_ohe_carr(train_df, 'UniqueCarrier')
code_ohe_carr(test_df, 'UniqueCarrier')
print("Add Dest features...")
code_ohe_dest(train_df, 'Dest')
code_ohe_dest(test_df, 'Dest')
print("Add Origin features...")
code_ohe_origin(train_df, 'Origin')
code_ohe_origin(test_df, 'Origin')
print(train_df.shape)
print(test_df.shape)
columns = test_df.columns.values.tolist()
columns.remove('Month')
columns.remove('DayofMonth')
columns.remove('DayOfWeek')
columns.remove('UniqueCarrier')
columns.remove('Origin')
columns.remove('Dest')
print("Split train/test parts...")
X_train = train_df[columns].values
X_test = test_df[columns].values
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=17)
print("Fit and predict...")
logit_pipe = Pipeline([('scaler', StandardScaler()), ('logit', LogisticRegression(C=1, random_state=17, solver='liblinear'))])
logit_pipe.fit(X_train_part, y_train_part)
logit_valid_pred = logit_pipe.predict_proba(X_valid)[:, 1]
print("ROC AUC NEW:"+str(roc_auc_score(y_valid, logit_valid_pred)))
print("Fit,predict, write file...")
logit_pipe.fit(X_train, y_train)
logit_test_pred = logit_pipe.predict_proba(X_test)[:, 1]
pd.Series(logit_test_pred, name='dep_delayed_15min').to_csv('logit_new_feat.csv', index_label='id', header=True)