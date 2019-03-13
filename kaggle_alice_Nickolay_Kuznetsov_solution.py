import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#Author - Nickolay Kuznetsov

# A helper function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file, target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels, index = np.arange(1, predicted_labels.shape[0] + 1), columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

times = ['time%s' % i for i in range(1, 11)]
sites = ['site%s' % i for i in range(1, 11)]
#read csv files
print("Read csv files...")
train_df = pd.read_csv('../input/train_sessions.csv', index_col='session_id', parse_dates=times)
test_df = pd.read_csv('../input/test_sessions.csv', index_col='session_id', parse_dates=times)
train_df = train_df.sort_values(by='time1')
train_df[sites].fillna(0).astype('int').to_csv('train_sessions_text.txt', sep=' ', index=None, header=None)
test_df[sites].fillna(0).astype('int').to_csv('test_sessions_text.txt', sep=' ', index=None, header=None)

#TF-IDF
print("TfidfVectorizer...")
cv = TfidfVectorizer(max_features=100000, ngram_range=(1, 3))
with open('train_sessions_text.txt') as inp_train_file:
    X_train = cv.fit_transform(inp_train_file)
with open('test_sessions_text.txt') as inp_test_file:
    X_test = cv.transform(inp_test_file)

#Save train targets into a separate vector.
y_train = train_df['target'].astype('int').values
#We'll be performing time series cross-validation, see sklearn TimeSeriesSplit and this dicussion on StackOverflow
time_split = TimeSeriesSplit(n_splits=10)
#Perform time series cross-validation with logistic regression
logit = LogisticRegression(C=1, random_state=17, solver='liblinear')

#Add  features
print("Make features...")
hour_train = train_df['time1'].apply(lambda ts: ts.hour)
morning_train = ((hour_train >= 7) & (hour_train <= 11)).astype('int')
day_train = ((hour_train >= 12) & (hour_train <= 18)).astype('int')
evening_train = ((hour_train >= 19) & (hour_train <= 23)).astype('int')
night_train = ((hour_train >= 0) & (hour_train <= 6)).astype('int')

morning_train = morning_train.values.reshape(-1,1)
day_train = day_train.values.reshape(-1,1)
evening_train = evening_train.values.reshape(-1,1)
night_train = night_train.values.reshape(-1,1)

hour_test = test_df['time1'].apply(lambda ts: ts.hour)
morning_test = ((hour_test >= 7) & (hour_test <= 11)).astype('int')
day_test = ((hour_test >= 12) & (hour_test <= 18)).astype('int')
evening_test = ((hour_test >= 19) & (hour_test <= 23)).astype('int')
night_test = ((hour_test >= 0) & (hour_test <= 6)).astype('int')

morning_test = morning_test.values.reshape(-1,1)
day_test = day_test.values.reshape(-1,1)
evening_test = evening_test.values.reshape(-1,1)
night_test = night_test.values.reshape(-1,1)

start_month_train = train_df['time1'].apply(lambda ts:100 * ts.year + ts.month).astype('float64')
start_month_train = StandardScaler().fit_transform(start_month_train.values.reshape(-1,1))
start_month_test = test_df['time1'].apply(lambda ts:100 * ts.year + ts.month).astype('float64')
start_month_test = StandardScaler().fit_transform(start_month_test.values.reshape(-1,1))

duration_train = train_df[times].max(axis=1)-train_df[times].min(axis=1)
duration_test = test_df[times].max(axis=1)-test_df[times].min(axis=1)
duration_train = duration_train.apply(lambda ts:ts.seconds).astype('float64')
duration_test = duration_test.apply(lambda ts:ts.seconds).astype('float64')
duration_train = StandardScaler().fit_transform(duration_train.values.reshape(-1,1))
duration_test = StandardScaler().fit_transform(duration_test.values.reshape(-1,1))

online_day_train = train_df['time1'].apply(lambda ts: ts.dayofweek in [0,1,3,4]).astype('float64')
online_day_test = test_df['time1'].apply(lambda ts: ts.dayofweek in [0,1,3,4]).astype('float64')
online_day_train = online_day_train.values.reshape(-1,1)
online_day_test = online_day_test.values.reshape(-1,1)


print("Add features...")
X_train_mod = hstack([X_train, morning_train, day_train, evening_train, night_train, start_month_train, duration_train, online_day_train])
X_test_mod = hstack([X_test, morning_test, day_test, evening_test, night_test, start_month_test, duration_test, online_day_test])

#Performing time series cross-validation, we see an improvement in ROC AUC.
print("Cross-validation...")
cv_scores = cross_val_score(logit, X_train_mod, y_train, cv=time_split, scoring='roc_auc', n_jobs=-1) 
#training
print("Training...")
logit.fit(X_train_mod, y_train)
#training
print("Predicting...")
logit_test_pred2 = logit.predict_proba(X_test_mod)[:, 1]
print("Write to file...")
write_to_submission_file(logit_test_pred2, 'submission_alice_Nickolay_Kuznetsov.csv') # 0.95469