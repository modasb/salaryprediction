# importer les librairies
import numpy as np
import pandas as pd

# %%
Data = pd.read_csv(r'C:\Users\ousse\Downloads\wetransfer_ml-model-deployment-using-flask-main_2024-05-10_1348\ML-MODEL-DEPLOYMENT-USING-FLASK-main\jobs.csv')
Data.duplicated().sum()
Data.drop_duplicates(inplace=True)
Data.drop(['salary', 'work_year'], axis=1, inplace=True)

# %%
Data

# %%
data = Data.copy()

# %%

# Count the occurrences of each job title
job_title_counts = data['job_title'].value_counts()

# Identify job titles with 3 or fewer occurrences
job_title_cat_other = job_title_counts[job_title_counts <= 3].index

# Replace rare job titles with 'other'
data['job_title'] = data['job_title'].apply(lambda x: 'other' if x in job_title_cat_other else x)

# Count the occurrences of each employee residence
emp_res_counts = data['employee_residence'].value_counts()

# Identify employee residences with 2 or fewer occurrences
emp_res_cat_other = emp_res_counts[emp_res_counts <= 2].index

# Replace rare employee residences with 'other'
data['employee_residence'] = data['employee_residence'].apply(lambda x: 'other' if x in emp_res_cat_other else x)

# Count the occurrences of each company location
com_loc_counts = data['company_location'].value_counts()

# Identify company locations with 2 or fewer occurrences
com_loc_cat_other = com_loc_counts[com_loc_counts <= 2].index

# Replace rare company locations with 'other'
data['company_location'] = data['company_location'].apply(lambda x: 'other' if x in com_loc_cat_other else x)

# %%

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

train_data, test_data = train_test_split(data, test_size=0.2, random_state=38)
train = data.drop('salary_in_usd', axis=1)
train_labels = data['salary_in_usd'].copy()

pipeline = Pipeline([
    ('one_hot_cat', OneHotEncoder())
])

train_pipelined = pipeline.fit_transform(train)
train_pipelined.shape
test = test_data.drop('salary_in_usd', axis=1)
test_labels = test_data['salary_in_usd'].copy()
test_pipelined = pipeline.transform(test)


# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Create DecisionTreeRegressor model
dt_model = DecisionTreeRegressor()

# Fit the model to the training data
dt_model.fit(train_pipelined, train_labels)

# Make predictions on the test data
dt_pred = dt_model.predict(test_pipelined)

# Calculer les métriques d'évaluation
dt_r2 = r2_score(test_labels, dt_pred)
dt_mse = mean_squared_error(test_labels, dt_pred)
dt_mae = mean_absolute_error(test_labels, dt_pred)

print(f"Score R² pour DecisionTreeRegressor : {dt_r2}")
print(f"Mean Squared Error pour DecisionTreeRegressor : {dt_mse}")
print(f"Mean Absolute Error pour DecisionTreeRegressor : {dt_mae}")
# %%
import matplotlib.pyplot as plt
import seaborn as sns
# Créer un scatter plot des valeurs prédites vs les valeurs réelles avec une ligne de régression
plt.figure(figsize=(22, 8))
sns.scatterplot(x=test_labels, y=dt_pred, color='b', alpha=0.9, edgecolor='k', s=80)
sns.regplot(x=test_labels, y=dt_pred, scatter=False, color='r', line_kws={"color": "red", "lw": 2})

plt.xlabel("Actual Values", fontsize=22)
plt.ylabel("Predicted Values", fontsize=22)
plt.title("Model Performance - Actual vs. Predicted Values", fontsize=22)

plt.show()
# %%
from joblib import dump

# Sauvegarder le modèle
from joblib import dump

# Assuming 'pipeline' is your preprocessing pipeline and 'dt_model' is your trained model
dump(pipeline, 'pipeline.pkl')
dump(dt_model, 'modele_regression.pkl')