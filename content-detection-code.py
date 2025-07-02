# Models

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report

# EDA

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# כללי

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
file_id = '1LKYuNkkby7fvLnxBmfeRnnBVsoXNdJPe'
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url)
df.info()
df.nlargest(10, 'track_popularity')
df[df['artist_name'] == 'Harry Styles'].sort_values(by='track_popularity', ascending=False).head(10)
df.groupby("artist_name")['followers'].max().nlargest(10)
under_15s_count = len(df[df['duration_ms'] < 15000])
over_1h_count = len(df[df['duration_ms'] > 3600000])

if under_15s_count > over_1h_count:
    print(f'There are more songs under 15s than songs over 1h. {under_15s_count} songs under 15 seconds.')
else:
    print(f'There are more songs over 1h than songs under 15s. {over_1h_count} songs over an hour long.')
decades = {1950 :0,
           1960:0,
           1970:0,
           1980:0,
           1990:0,
           2000:0,
           2010:0}
for i in df.index:
  if df.loc[i,"explicit"]:
      value = df.loc[i, "release_date"]
      year = float(str(value)[:4])


      if (((1950 <= year) & ( year <= 1959))):
        decades[1950] += 1
      if (((1960 <= year) & ( year <= 1969))):
        decades[1960] += 1
      if (((1970 <= year) & ( year <= 1979))):
        decades[1970] += 1
      if (((1980 <= year) & ( year <= 1989))):
        decades[1980] += 1
      if (((1990 <= year) & ( year <= 1999))):
        decades[1990] += 1
      if (((2000 <= year) & ( year <= 2009))):
        decades[2000] += 1
      if (((2010 <= year) & ( year <= 2019))):
        decades[2010] += 1

decades_keys = list(decades.keys())
decades_values = list(decades.values())
plt.plot(decades_keys, decades_values, marker = "o")
plt.grid(True)
plt.show() 

sample_df = df.sample(n=50000, random_state=42)

plt.figure(figsize=(10, 6))
sns.regplot(x='danceability', y='track_popularity', data=sample_df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Danceability vs. Popularity with Trend Line')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.show()
file_id = '1lJZi6o9TBVIKmDwOe9GOYegKinP9wxtn'
url = f"https://drive.google.com/uc?id={file_id}"
EDA_df = pd.read_csv(url)
EDA_df.info()
EDA_df.isna().sum()
EDA_df.dropna(subset = ["danceability", "energy", "release_date"], inplace = True)
EDA_df['release_date'].fillna('Unknown', inplace=True)
EDA_df['artist_name'].fillna('Unknown', inplace=True)
EDA_df['track_name'].fillna('Unknown', inplace=True)
EDA_df["release_date"] = pd.to_datetime(EDA_df["release_date"])
EDA_df["release_year"] =EDA_df["release_date"].apply(lambda x: int(str(x)[:4]))
EDA_df.drop(['artist_id', 'type', 'track_name', 'artist_name', 'release_date'], axis=1, inplace=True)
EDA_df["explicit"] = EDA_df["explicit"].apply(lambda x: 0 if x == True else 1)
zero = EDA_df[EDA_df['num_beats'] == 'Zero']
zero_beats = zero.index
EDA_df.drop(zero_beats, inplace = True)
EDA_df["num_beats"].value_counts()

X = EDA_df.drop(['explicit'], axis=1)
y = EDA_df['explicit']
from sklearn.model_selection import KFold
knn = KNeighborsClassifier(n_neighbors= 10)
t = DecisionTreeClassifier()
K = 10
kf = KFold(n_splits=K, random_state=42, shuffle= True)
scoring_method = ["accuracy", "precision", "recall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42)
cv_results_t = cross_validate(estimator = t, X = X_train, y = y_train, cv = kf, scoring = scoring_method)
cv_results_knn = cross_validate(estimator = knn, X = X_train, y = y_train, cv = kf, scoring = scoring_method)
print("Decision Tree: ")
print("Accuracy: ", cv_results_t["test_accuracy"].mean())
print("Precision: ", cv_results_t["test_precision"].mean())
print("Recall: ", cv_results_t["test_recall"].mean())
print("KNN: ")
print("Accuracy: ", cv_results_knn["test_accuracy"].mean())
print("Precision: ", cv_results_knn["test_precision"].mean())
print("Recall: ", cv_results_knn["test_recall"].mean())
def iqr_remove(X,y,column):
  q1 = X[column].quantile(0.25)
  q3 = X[column].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - (1.5 * iqr)
  upper_bound = q3 + (1.5 * iqr)

  indices_to_drop = (X[column] < lower_bound) | (X[column] > upper_bound)
  X.drop(X[indices_to_drop].index, inplace=True)
  y.drop(y[indices_to_drop].index, inplace=True)
X['danceability'].describe()
sns.boxplot(X['danceability'])
plt.show()
iqr_remove(X, y, 'danceability')
sns.boxplot(X['danceability'])
sns.histplot(X['danceability'])

scaler = StandardScaler()
X['danceability'] = scaler.fit_transform(X[['danceability']])
EDA_df.corr()
def count_outliers(df, threshold=15):
    outliers = {}

    for column in df.columns:
        if df[column].dtype not in [np.float64, np.int64]:
            continue
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        num_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        outliers[column] = num_outliers

    return outliers
count_outliers(EDA_df)
sns.boxplot(X['danceability'])
iqr_remove(X, y, 'danceability')
sns.boxplot(X['danceability'])
sns.histplot(X['danceability'])
sns.boxplot(X["duration_ms"])
iqr_remove(X, y, "duration_ms")
sns.boxplot(X["duration_ms"])
sns.histplot(X["duration_ms"])
X['duration_ms'] = scaler.fit_transform(X[['duration_ms']])
sns.boxplot(X["key"])
iqr_remove(X, y, "key")
sns.boxplot(X["key"])
sns.histplot(X["key"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42)
cv_results_t = cross_validate(estimator = t, X = X_train, y = y_train, cv = kf, scoring = scoring_method)
cv_results_knn = cross_validate(estimator = knn, X = X_train, y = y_train, cv = kf, scoring = scoring_method)
print("Decision Tree: ")
print("Accuracy: ", cv_results_t["test_accuracy"].mean())
print("Precision: ", cv_results_t["test_precision"].mean())
print("Recall: ", cv_results_t["test_recall"].mean())
print("KNN: ")
print("Accuracy: ", cv_results_knn["test_accuracy"].mean())
print("Precision: ", cv_results_knn["test_precision"].mean())
print("Recall: ", cv_results_knn["test_recall"].mean())
count_outliers(X)

sns.boxplot(X["tempo"])
iqr_remove(X, y, "tempo")
sns.boxplot(X["tempo"])
sns.histplot(X["tempo"])
X["tempo"] = scaler.fit_transform(X[["tempo"]])
for col in X.columns:
    plt.figure(figsize=(10, 4))

    # BoxPlot
    plt.subplot(1, 2, 1)
    sns.boxplot(x=X[col])
    plt.title(f'Box Plot of {col}')

    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(x=X[col], bins=20, kde=True)
    plt.title(f'Histogram of {col}')

    plt.tight_layout()
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42)
cv_results_t = cross_validate(estimator = t, X = X_train, y = y_train, cv = kf, scoring = scoring_method)
cv_results_knn = cross_validate(estimator = knn, X = X_train, y = y_train, cv = kf, scoring = scoring_method)
print("Decision Tree: ")
print("Accuracy: ", cv_results_t["test_accuracy"].mean())
print("Precision: ", cv_results_t["test_precision"].mean())
print("Recall: ", cv_results_t["test_recall"].mean())
print("KNN: ")
print("Accuracy: ", cv_results_knn["test_accuracy"].mean())
print("Precision: ", cv_results_knn["test_precision"].mean())
print("Recall: ", cv_results_knn["test_recall"].mean())
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
print('Distribution of the oversampled target variable:', Counter(y_resampled))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42)
cv_results_t = cross_validate(estimator = t, X = X_train, y = y_train, cv = kf, scoring = scoring_method)
cv_results_knn = cross_validate(estimator = knn, X = X_train, y = y_train, cv = kf, scoring = scoring_method)
print("Decision Tree: ")
print("Accuracy: ", cv_results_t["test_accuracy"].mean())
print("Precision: ", cv_results_t["test_precision"].mean())
print("Recall: ", cv_results_t["test_recall"].mean())
print("KNN: ")
print("Accuracy: ", cv_results_knn["test_accuracy"].mean())
print("Precision: ", cv_results_knn["test_precision"].mean())
print("Recall: ", cv_results_knn["test_recall"].mean())
from sklearn.neighbors import NearestCentroid as NCC

ncc_model = NCC()
ncc_model.fit(X_train, y_train)
ncc_pred = ncc_model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, ncc_pred))
print('Recall: ', recall_score(y_test, ncc_pred))
print('Precision: ', precision_score(y_test, ncc_pred))
knn_param_grid = {'n_neighbors': range(1, 6)}
dt_param_grid = {'max_depth': [None, 5, 10]}
rf_param_grid = {'n_estimators': [5, 10]}

knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=kf, scoring='recall')
dt_grid_search = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=kf, scoring='recall')
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=kf, scoring='recall')

knn_grid_search.fit(X_resampled, y_resampled)
dt_grid_search.fit(X_resampled, y_resampled)
rf_grid_search.fit(X_resampled, y_resampled)

print("KNN - Best parameters:", knn_grid_search.best_params_)
print("KNN - Best score:", knn_grid_search.best_score_)
print("Decision Tree - Best parameters:", dt_grid_search.best_params_)
print("Decision Tree - Best score:", dt_grid_search.best_score_)
print("Random Forest - Best parameters:", rf_grid_search.best_params_)
print("Random Forest - Best score:", rf_grid_search.best_score_)

knn_results = knn_grid_search.cv_results_
plt.figure()
plt.plot(knn_results['param_n_neighbors'], knn_results['mean_test_score'], marker='o')
plt.title("KNN Performance for Different Values of K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Mean Test Score")
plt.grid(True)
plt.show()
knn_bagging_clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=5, random_state=42)

knn_bagging_clf.fit(X_resampled, y_resampled)
knn_bagging_pred = knn_bagging_clf.predict(X_test)

knn_bagging_precision = precision_score(y_test, knn_bagging_pred)
knn_bagging_recall = recall_score(y_test, knn_bagging_pred)
knn_bagging_accuracy = accuracy_score(y_test, knn_bagging_pred)

print("KNN Bagging Classifier - Precision:", knn_bagging_precision)
print("KNN Bagging Classifier - Recall:", knn_bagging_recall)
print("KNN Bagging Classifier - Accuracy:", knn_bagging_accuracy)


dt_bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=10, random_state=42)
dt_bagging_clf.fit(X_resampled, y_resampled)
dt_bagging_pred = dt_bagging_clf.predict(X_test)

dt_bagging_precision = precision_score(y_test, dt_bagging_pred)
dt_bagging_recall = recall_score(y_test, dt_bagging_pred)
dt_bagging_accuracy = accuracy_score(y_test, dt_bagging_pred)

print("Decision Tree Bagging Classifier - Precision:", dt_bagging_precision)
print("Decision Tree Bagging Classifier - Recall:", dt_bagging_recall)
print("Decision Tree Bagging Classifier - Accuracy:", dt_bagging_accuracy)

# Clustering Section

file_id = '1PWKkenkH4HJFygarpfLfq97HchCKp-Rn'
url = f"https://drive.google.com/uc?id={file_id}"
data_cluster = pd.read_csv(url, encoding='iso-8859-1')
data_cluster.info()

data_cluster.drop(['track_name'], axis=1, inplace=True)

for column in data_cluster.columns:
    scaler = MinMaxScaler()
    data_cluster[column] = scaler.fit_transform(data_cluster[[column]])
ssd = []
k_inertia = range(1, 11)

for k in k_inertia:
    kmeans = KMeans(n_clusters=k, n_init = 'auto')
    kmeans.fit(data_cluster)
    ssd.append(kmeans.inertia_)

plt.plot(k_inertia, ssd, marker='o')
plt.title('Elbow Point Graph')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.xticks(k_inertia)
plt.grid(True)
plt.show()
k_inertia = pd.DataFrame(columns = ['inertia_values', 'k_values'])
for k in range(1, 16):
    kmeans2 = KMeans(n_clusters = k, n_init = 'auto')
    kmeans2.fit(data_cluster)
    inertia = kmeans2.inertia_
    k_inertia = k_inertia.append({'k_values': k, 'inertia_values': inertia}, ignore_index=True)

sns.lineplot(data = k_inertia, x = "k_values", y = "inertia_values")
kmeans = KMeans(n_clusters = 3, n_init = 'auto')
kmeans.fit(data_cluster)
data_cluster['cluster'] = kmeans.labels_
print("inertia result:", kmeans.inertia_)
data_cluster.groupby('cluster').mean()
data_cluster["cluster"].value_counts()

palette = ["blue", "yellow", "green", "purple", "red", "orange"]
sns.scatterplot(data = data_cluster, x = 'danceability_%', y = 'valence_%', hue = "cluster", palette=palette)
plt.xlabel('danceability')
plt.ylabel('valence')
plt.legend()
plt.show()
data_cluster['edm'] = data_cluster[data_cluster['valence_%'].between(0.6, 0.99)]

data_cluster['metal'] = data_cluster[data_cluster['energy_%'].between(0.8, 0.99)]
data_cluster['rap'] = data_cluster[data_cluster['speechiness_%'].between(0.7, 0.99)]
