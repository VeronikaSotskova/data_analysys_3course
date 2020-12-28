import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

def get_df(samples):
  df = pd.read_csv('/content/drive/MyDrive/Анализ_Данных_Соцкова_Вероника_11-802/Practice/balanced_utkonos.csv', low_memory=False)
  df.index = df['Unnamed: 0'].values
  df = df.drop('Unnamed: 0', axis=1)
  df = df.dropna()
  df_0 = df[df.CancelFlag==0].sample(samples)
  df_1 = df[df.CancelFlag==1].sample(samples)
  df = pd.concat([df_0, df_1])

  return df

def data_preprocessing(df):
  tmp = df['Interval'].str.split('-')
  df['interval_low']=tmp.apply(lambda x: int(x[0]))
  df['interval_high']=tmp.apply(lambda x: int(x[1][:-1]))
  del df['Interval']

  df['interval_avg'] = (df['interval_high'] + df['interval_low']) / 2

  morning = list(range(6, 12))
  day = list(range(12, 18))
  evening = list(range(18, 24))
  night = [24] + list(range(1, 6))

  morning = df['interval_avg'].isin(morning)
  day = df['interval_avg'].isin(day)
  evening = df['interval_avg'].isin(evening)
  night = df['interval_avg'].isin(night)

  df['morning'] = morning.apply(lambda x: 1 if x else 0)
  df['day'] = day.apply(lambda x: 1 if x else 0)
  df['evening'] = evening.apply(lambda x: 1 if x else 0)
  df['night'] = night.apply(lambda x: 1 if x else 0)

  orderDate = df.OrderDate.apply (lambda x: datetime.datetime.strptime (x, '%d/%m/%Y'))
  date = df.Date.apply (lambda x: datetime.datetime.strptime (x, '%d/%m/%Y'))
  df['delta_day'] = (date-orderDate).dt.days.astype(int).values

  df['count_edit'] -= 1

  df['DeliveryType'] = df['DeliveryType'].map({'Обычная доставка': 0, 'Доставка День в День': 1})

  df = df.drop(['ClientID'], axis=1)

  del df['Date']
  del df['OrderDate']
  del df['OrderID']

  le = LabelEncoder()
  df['Cluster'] = le.fit_transform(df['Cluster'])

  return df

def get_pies(df):
  #1
  cancel_data = df[['CancelFlag', 'morning',	'day', 'evening',	'night']].groupby('CancelFlag').sum()
  size1 = cancel_data.values[1]
  size2 = cancel_data.values[0]
  proportion=size1/(size1+size2)
  labels = 'morning',	'day', 'evening',	'night'
  fig1 = px.pie(cancel_data, values=proportion*100, names=labels)

  #2
  cancel_data2 = df[df.CancelFlag==1].groupby('prepay').count()
  labels2 = 'Отказывают без предоплаты', 'Отказывают с предоплатой'

  fig2 = px.pie(cancel_data2, values=cancel_data2['CancelFlag'].values, names=labels2)
  return fig1, fig2

def get_X_y(df):
  X = df.drop('CancelFlag', axis=1)
  y = df.CancelFlag.astype(int)  

  return X, y

def get_K_best_features(X, y):
  selector = SelectKBest(f_classif, k=10)
  selector.fit(X, y)

  X_best_col = X.columns[selector.get_support(indices=True)]
  return X_best_col

def get_train_test_scaled(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
  scaler = StandardScaler()
  scaler.fit(X_train)

  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  return X_train_scaled, X_test_scaled, y_train, y_test

def get_roc_auc(model, X_test, y_test):
  y_score = model.predict_proba(X_test)[:, 1]

  fpr, tpr, thresholds = roc_curve(y_test, y_score)

  fig = px.area(
      x=fpr, y=tpr,
      title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
      labels=dict(x='False Positive Rate', y='True Positive Rate'),
      width=700, height=500
  )
  fig.add_shape(
      type='line', line=dict(dash='dash'),
      x0=0, x1=1, y0=0, y1=1
  )

  fig.update_yaxes(scaleanchor="x", scaleratio=1)
  fig.update_xaxes(constrain='domain')
  return fig

def get_knn_model(X_train, y_train):
  knn = KNeighborsClassifier(n_neighbors=25)
  knn.fit(X_train, y_train)
  return knn

def get_decision_tree(X_train, y_train):
  tree_param = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
  tree_gridsearch = GridSearchCV(DecisionTreeClassifier(), tree_param, cv=5)
  tree_gridsearch.fit(X_train, y_train)
  return tree_gridsearch

def get_random_forest(X_train, y_train):
  rand_forest = RandomForestClassifier(max_depth=2, random_state=0)
  rand_forest.fit(X_train, y_train)
  return rand_forest

def get_bagging(X_train, y_train):
  b_model = BaggingClassifier()
  b_model.fit(X_train,y_train)
  return b_model