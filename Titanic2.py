import pandas as pd
import numpy as np
import re
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

def preprocess_data(data):
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""

    data['Title'] = data['Name'].apply(get_title)
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
    
    data['Age*Class'] = data['Age'] * data['Pclass']

    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Sex'] = data['Sex'].map({"male": 0, "female": 1}).astype(int)
    data['Embarked'] = data['Embarked'].map({"S": 0, "C": 1, "Q": 2}).astype(int)

    age_df = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    train_age = data.loc[data.Age.notna()]
    test_age = data.loc[data.Age.isna()]
    y = train_age.values[:, 0]
    X = train_age.values[:, 1:]
    rfr = RandomForestRegressor(n_jobs=-1)
    rfr.fit(X, y)
    predicted_ages = rfr.predict(test_age.values[:, 1:])
    data.loc[data.Age.isna(), 'Age'] = predicted_ages

    return data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

X = train_data.drop("Survived", axis=1)
Y = train_data["Survived"]

stratified = StratifiedKFold(n_splits=10)

model = XGBClassifier(use_label_encoder=False)

param_grid = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=stratified)

grid_result = grid_search.fit(X, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model = XGBClassifier(
    use_label_encoder=False,
    min_child_weight=grid_result.best_params_["min_child_weight"],
    gamma=grid_result.best_params_["gamma"],
    subsample=grid_result.best_params_["subsample"],
    colsample_bytree=grid_result.best_params_["colsample_bytree"],
    max_depth=grid_result.best_params_["max_depth"]
)
model.fit(X, Y)

predictions = model.predict(test_data)

output = pd.DataFrame({'PassengerId': np.arange(892, 892+len(test_data)), 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
