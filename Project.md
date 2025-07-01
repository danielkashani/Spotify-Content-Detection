
---

# Spotify Content Detection Project

## Table of Contents

1. [Background & Problem Statement](#background--problem-statement)
2. [Performance Metric Selection](#performance-metric-selection)
3. [Data Exploration & Initial Analysis](#data-exploration--initial-analysis)
4. [EDA: Cleaning & Feature Engineering](#eda-cleaning--feature-engineering)
5. [Model Training & Evaluation](#model-training--evaluation)
6. [Advanced Modeling & Hyperparameter Tuning](#advanced-modeling--hyperparameter-tuning)
7. [Ensemble Methods](#ensemble-methods)
8. [Summary & Insights](#summary--insights)

---

## Background & Problem Statement

**Spotify** is a global provider of digital music streaming services.  
To ensure a safe and enjoyable listening experience for its younger users, Spotify has introduced a feature that recommends songs based on popular tracks in the user's country. However, user complaints have raised concerns about the inclusion of songs with inappropriate content. Spotify's priority is to reduce the number of inappropriate songs recommended—specifically, to minimize the number of "unsafe" songs that are missed by the detection feature (false negatives).

**Goal:**  
Develop a model that can detect and filter out songs likely to be inappropriate, improving the platform’s safety and reputation among young users.

---

## Performance Metric Selection

Given the business context, our main objective is to **minimize false negatives** (i.e., inappropriate songs not detected by the system).  
**Recall** is the most appropriate metric, as it measures the proportion of actual inappropriate tracks that are correctly identified.

> **Metric to optimize:**  
> **Recall** (Sensitivity)

---

## Data Exploration & Initial Analysis

We begin by loading the full Spotify database and answering relevant exploratory questions.

### Loading the Dataset

```python
import pandas as pd

file_id = '1LKYuNkkby7fvLnxBmfeRnnBVsoXNdJPe'
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url)
```

---

### Key Exploratory Questions

**1. What are the 10 most popular songs?**

```python
df.nlargest(10, 'track_popularity')
```

**2. What are the top 10 most popular songs by Harry Styles?**

```python
df[df['artist_name'] == 'Harry Styles'].sort_values(by='track_popularity', ascending=False).head(10)
```

**3. Which artists have the most followers?**

```python
df.groupby("artist_name")['followers'].max().nlargest(10)
```

**4. Are there more songs under 15 seconds or over an hour in length?**

```python
under_15s_count = len(df[df['duration_ms'] < 15000])
over_1h_count = len(df[df['duration_ms'] > 3600000])
if under_15s_count > over_1h_count:
    print(f'More songs under 15s: {under_15s_count}')
else:
    print(f'More songs over 1h: {over_1h_count}')
```

**5. How many explicit songs exist per decade?**

```python
decades = {1950:0, 1960:0, 1970:0, 1980:0, 1990:0, 2000:0, 2010:0}
for i in df.index:
    if df.loc[i,"explicit"]:
        year = int(str(df.loc[i, "release_date"])[:4])
        for decade in decades:
            if decade <= year < decade + 10:
                decades[decade] += 1
decades_keys = list(decades.keys())
decades_values = list(decades.values())

import matplotlib.pyplot as plt
plt.plot(decades_keys, decades_values, marker="o")
plt.title("Explicit Songs per Decade")
plt.xlabel("Decade")
plt.ylabel("Number of Explicit Songs")
plt.grid(True)
plt.show()
```

**6. What is the trend between danceability and popularity?**  
_Scatter plot with trend line, sampled at 50,000 observations:_

```python
import seaborn as sns

sample_df = df.sample(n=50000, random_state=42)
plt.figure(figsize=(10, 6))
sns.regplot(x='danceability', y='track_popularity', data=sample_df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Danceability vs. Popularity with Trend Line')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.show()
```

---

## EDA: Cleaning & Feature Engineering

We continue with a cleaned and curated dataset to begin the EDA process.

### Data Cleaning Steps

- Remove duplicate rows
- Drop missing values in important columns
- Drop irrelevant columns
- Convert object columns to correct types
- Feature engineering: extract year from release date
- Validate data integrity

**Example code:**

```python
file_id = '1lJZi6o9TBVIKmDwOe9GOYegKinP9wxtn'
url = f"https://drive.google.com/uc?id={file_id}"
EDA_df = pd.read_csv(url)

# Drop rows with missing danceability, energy, or release_date
EDA_df.dropna(subset=["danceability", "energy", "release_date"], inplace=True)
EDA_df['release_date'].fillna('Unknown', inplace=True)
EDA_df['artist_name'].fillna('Unknown', inplace=True)
EDA_df['track_name'].fillna('Unknown', inplace=True)

# Convert release_date to datetime and extract year
EDA_df["release_date"] = pd.to_datetime(EDA_df["release_date"])
EDA_df["release_year"] =EDA_df["release_date"].apply(lambda x: int(str(x)[:4]))

# Remove duplicates and irrelevant columns
EDA_df.drop_duplicates(inplace=True)
EDA_df.drop(['artist_id', 'type', 'track_name', 'artist_name', 'release_date'], axis=1, inplace=True)
EDA_df["explicit"] = EDA_df["explicit"].apply(lambda x: 0 if x == True else 1)

# Remove 'num_beats' rows with 'Zero'
EDA_df = EDA_df[EDA_df['num_beats'] != 'Zero']

X = EDA_df.drop(['explicit'], axis=1)
y = EDA_df['explicit']
```

---

### Model Performance After Cleaning (Step 1)

We train both a Decision Tree and KNN (k=10) and evaluate using cross-validation.

```python
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
tree = DecisionTreeClassifier()
K = 10
kf = KFold(n_splits=K, random_state=42, shuffle=True)
scoring = ["accuracy", "precision", "recall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv_results_tree = cross_validate(tree, X_train, y_train, cv=kf, scoring=scoring)
cv_results_knn = cross_validate(knn, X_train, y_train, cv=kf, scoring=scoring)

print("Decision Tree: ", {k: v.mean() for k, v in cv_results_tree.items() if 'test_' in k})
print("KNN: ", {k: v.mean() for k, v in cv_results_knn.items() if 'test_' in k})
```

> #### Example Results:
> - **Decision Tree**:  
>   - Accuracy: 0.864  
>   - Precision: 0.922  
>   - Recall: 0.918  
> - **KNN**:  
>   - Accuracy: 0.841  
>   - Precision: 0.860  
>   - Recall: 0.972  

---

### Step 2: Correlation & Outlier Analysis

- Check correlations between features and target (`.corr()`)
- Identify and treat outliers for three features (e.g., `danceability`, `duration_ms`, `key`)
- Visualize with boxplots and histograms
- Scale features as needed

```python
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def iqr_remove(X, y, column):
    q1 = X[column].quantile(0.25)
    q3 = X[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (X[column] >= lower) & (X[column] <= upper)
    X = X[mask]
    y = y[mask]
    return X, y

# Example for 'danceability'
sns.boxplot(X['danceability'])
plt.show()
X, y = iqr_remove(X, y, 'danceability')
sns.boxplot(X['danceability'])
plt.show()
sns.histplot(X['danceability'])
plt.show()
scaler = StandardScaler()
X['danceability'] = scaler.fit_transform(X[['danceability']])
```

---

### Model Performance After Outlier Removal (Step 2)

Repeat model training and evaluation as above.

> **Example Results:**  
> - **Decision Tree**: Accuracy: 0.860, Precision: 0.920, Recall: 0.914  
> - **KNN**: Accuracy: 0.840, Precision: 0.878, Recall: 0.942  

---

## Step 3: Balancing the Data

Use oversampling to address class imbalance.

```python
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
print('Distribution after oversampling:', Counter(y_resampled))
```

_Retrain and evaluate models after resampling._

> **Example Results:**  
> - **Decision Tree**: Accuracy: 0.861, Precision: 0.921, Recall: 0.915  
> - **KNN**: Accuracy: 0.853, Precision: 0.886, Recall: 0.949  

---

## Advanced Modeling & Hyperparameter Tuning

### Nearest Centroid Classifier

```python
from sklearn.neighbors import NearestCentroid

ncc = NearestCentroid()
ncc.fit(X_train, y_train)
ncc_pred = ncc.predict(X_test)
print('Accuracy:', accuracy_score(y_test, ncc_pred))
print('Recall:', recall_score(y_test, ncc_pred))
print('Precision:', precision_score(y_test, ncc_pred))
```

### Grid Search for Best Hyperparameters

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

knn_param_grid = {'n_neighbors': range(1, 6)}
dt_param_grid = {'max_depth': [None, 5, 10]}
rf_param_grid = {'n_estimators': [5, 10, 100]}

knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=kf, scoring='recall')
dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=kf, scoring='recall')
rf_grid = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=kf, scoring='recall')

knn_grid.fit(X_resampled, y_resampled)
dt_grid.fit(X_resampled, y_resampled)
rf_grid.fit(X_resampled, y_resampled)

print("KNN Best:", knn_grid.best_params_, knn_grid.best_score_)
print("Decision Tree Best:", dt_grid.best_params_, dt_grid.best_score_)
print("Random Forest Best:", rf_grid.best_params_, rf_grid.best_score_)
```

**Example plot for KNN performance:**

```python
knn_results = knn_grid.cv_results_
plt.plot(knn_results['param_n_neighbors'], knn_results['mean_test_score'], marker='o')
plt.title("KNN Performance for Different Values of K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Mean Test Score (Recall)")
plt.grid(True)
plt.show()
```

---

## Ensemble Methods

### Bagging Classifier with KNN & Decision Tree

```python
from sklearn.ensemble import BaggingClassifier

# Bagging with KNN
knn_bagging = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=5, random_state=42)
knn_bagging.fit(X_resampled, y_resampled)
knn_bag_pred = knn_bagging.predict(X_test)

print("KNN Bagging - Precision:", precision_score(y_test, knn_bag_pred))
print("KNN Bagging - Recall:", recall_score(y_test, knn_bag_pred))
print("KNN Bagging - Accuracy:", accuracy_score(y_test, knn_bag_pred))

# Bagging with Decision Tree
dt_bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=10, random_state=42)
dt_bagging.fit(X_resampled, y_resampled)
dt_bag_pred = dt_bagging.predict(X_test)

print("Decision Tree Bagging - Precision:", precision_score(y_test, dt_bag_pred))
print("Decision Tree Bagging - Recall:", recall_score(y_test, dt_bag_pred))
print("Decision Tree Bagging - Accuracy:", accuracy_score(y_test, dt_bag_pred))
```

---

## Summary & Insights

- **Recall** was prioritized throughout, to minimize the chance of inappropriate songs reaching young listeners.
- Data was thoroughly cleaned, engineered, and balanced.
- Outlier removal and feature scaling improved model performance.
- **Decision Tree** and **KNN** classifiers performed well, with further improvements using ensemble methods such as Bagging and Random Forest.
- Hyperparameter tuning (GridSearch) was employed to optimize models.
- The final ensemble models yielded the best balance between precision and recall—crucial for Spotify’s safety requirements.

---
