import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle

# Read Data
df = pd.read_csv('./data/final_data.csv')

# Split and Set Target: 60/20/20 split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train_orig = df_train['latestPrice'].to_numpy()
y_val_orig = df_val['latestPrice'].to_numpy()
y_test_orig = df_test['latestPrice'].to_numpy()

y_train = np.log1p(y_train_orig)
y_val = np.log1p(y_val_orig)
y_test = np.log1p(y_test_orig)

df_train = df_train.drop(columns='latestPrice')
df_val = df_val.drop(columns='latestPrice')
df_test = df_test.drop(columns='latestPrice')

# Vectorize
train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=True) # pickle
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

# Init and Train RF 
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=1) # pickle
rf.fit(X_train, y_train)

# Tune
n_estimators= list(np.arange(1,22,2))

# Set the parameters to train as well as their ranges
param_grid_forest = {"n_estimators": n_estimators, 
                    "n_jobs": [1,2,3,4,5] # testing n_jobs
                    }

rf_tuned = GridSearchCV(
                            rf, 
                            param_grid_forest, 
                            cv = 3, 
                            n_jobs = 1, 
                            verbose = 5
                        )

# Train on data, hypertuned based on gridsearch
rf_tuned.fit(X_train, y_train)

# Export model using pickle (Only if satisfied with the evaluation)
model_filename = 'rf_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(rf_tuned, file)

# Export vectorizer
dv_filename = 'dv.pkl'
with open(dv_filename, 'wb') as file:
    pickle.dump(dv, file)