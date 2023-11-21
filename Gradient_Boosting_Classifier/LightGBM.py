import numpy as np
import pandas as pd
import pickle5 as pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
# Reading the dataset
df = pd.read_csv('./data/dataset_20_10_23.csv')
# Splitting the data
X = df.drop(['id', 'class'], axis=1)  # Features excluding 'id' and 'class'
y = df['class']  # Target variable
class_counts = y.value_counts()
single_sample_classes = class_counts[class_counts == 1].index
filter_mask = ~y.isin(single_sample_classes)
X = X[filter_mask]
y = y[filter_mask]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Set up LGBMClassifier (this is the sklearn-style version of LightGBM)
model = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=17)
# Setting up grid of parameters to search
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [20, 50, 100, 200],
    'max_depth': [-1, 3, 5, 7]
}
# Create the grid search with 5-fold cross validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='precision_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
# Printing the best parameters from the grid search
print("Best parameters found: ", grid_search.best_params_)
print("Highest precision score found: ", grid_search.best_score_)
# Predicting using the best model
y_pred_test = grid_search.best_estimator_.predict(X_test)
y_pred_train = grid_search.best_estimator_.predict(X_train)
# Displaying precision scores
print('TEST = ', precision_score(y_pred_test, y_test, average='macro'))
print('TRAIN = ', precision_score(y_pred_train, y_train, average='macro'))
with open(f'lgbm_model_{precision_score(y_pred_test, y_test, average="macro")}.pkl', 'wb') as model_file:
    pickle.dump(grid_search.best_estimator_, model_file)