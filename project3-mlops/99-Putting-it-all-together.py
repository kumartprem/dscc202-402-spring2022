# Databricks notebook source
# MAGIC %md
# MAGIC # Putting it all together: Managing the Machine Learning Lifecycle
# MAGIC 
# MAGIC Create a workflow that includes pre-processing logic, the optimal ML algorithm and hyperparameters, and post-processing logic.
# MAGIC 
# MAGIC ## Instructions
# MAGIC 
# MAGIC In this course, we've primarily used Random Forest in `sklearn` to model the Airbnb dataset.  In this exercise, perform the following tasks:
# MAGIC <br><br>
# MAGIC 0. Create custom pre-processing logic to featurize the data
# MAGIC 0. Try a number of different algorithms and hyperparameters.  Choose the most performant solution
# MAGIC 0. Create related post-processing logic
# MAGIC 0. Package the results and execute it as its own run
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Web browser: Chrome
# MAGIC - A cluster configured with **8 cores** and **DBR 7.0 ML**

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the<br/>
# MAGIC start of each lesson (see the next cell) and the **`Classroom-Cleanup`** cell at the end of each lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# Adust our working directory from what DBFS sees to what python actually sees
working_path = workingDir.replace("dbfs:", "/dbfs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-processing
# MAGIC 
# MAGIC Take a look at the dataset and notice that there are plenty of strings and `NaN` values present. Our end goal is to train a sklearn regression model to predict the price of an airbnb listing.
# MAGIC 
# MAGIC 
# MAGIC Before we can start training, we need to pre-process our data to be compatible with sklearn models by making all features purely numerical. 

# COMMAND ----------

import pandas as pd

airbnbDF = spark.read.parquet("/mnt/training/airbnb/sf-listings/sf-listings-correct-types.parquet").toPandas()

display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cells we will walk you through the most basic pre-processing step necessary. Feel free to add additional steps afterwards to improve your model performance.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC First, convert the `price` from a string to a float since the regression model will be predicting numerical values.

# COMMAND ----------

airbnbDF['price'] = airbnbDF['price'].replace('[\$,]', '', regex=True).astype(float)
display(airbnbDF)

# COMMAND ----------

# Do univariate analysis of the numeric variables
num_cols = airbnbDF.select_dtypes(exclude='object').columns.tolist()
data_num = airbnbDF[num_cols]

cardinality_ = pd.DataFrame(data_num.nunique()).rename(columns={0:'Cardinality'})
mean_std_ = data_num.describe().transpose().drop(columns=['min','25%','50%','75%','max'])
quantiles_ = data_num.quantile([.0, .01, .05, .1, .25, .5, .75, .90, .95, .99, 1]).transpose().rename(columns={0.00:'Min', 0.01:'P01', 0.05:'P05', 0.10:'P10', 0.25:'P25', 0.50:'Median', 0.75:'P75', 0.90:'P90', 0.95:'P95', 0.99:'P99', 1.00:'Max'})

univariate_num = pd.concat([cardinality_, mean_std_, quantiles_], axis=1)

pd.DataFrame(airbnbDF.dtypes).rename(columns={0:'Type'})\
.join(pd.DataFrame(airbnbDF.isnull().sum()).rename(columns={0:'Null_count'}))\
.join(univariate_num, how='inner')

# COMMAND ----------

# Get the frequency distribution of the categorical variables
freq_dist = pd.DataFrame()
obj_cols = airbnbDF.select_dtypes(include='object').columns.tolist()
for col in obj_cols:
    test = pd.DataFrame(airbnbDF[col].value_counts()).reset_index().rename(columns={'index':'category'}).sort_values(col,0,ascending=False)
    test[col+'1'] = test['category'] + ":" + test[col].astype('str')
    test = test.drop(columns=['category',col]).rename(columns={col+'1':col})[0:10].transpose()
    freq_dist = pd.concat([freq_dist,test],axis=0,join="outer")

cardinality_ = pd.DataFrame(airbnbDF[obj_cols].nunique()).rename(columns={0:'Cardinality'})

pd.DataFrame(airbnbDF.dtypes).rename(columns={0:'Type'})\
.join(pd.DataFrame(airbnbDF.isnull().sum()).rename(columns={0:'Null_count'}))\
.join(cardinality_)\
.join(freq_dist, how='inner')

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at our remaining columns with strings (or numbers) and decide if you would like to keep them as features or not.
# MAGIC 
# MAGIC Remove the features you decide not to keep.

# COMMAND ----------

# Drop the zipcode variable, since we have latitude and longitude which are more granular
# Explainability wise, it's better to group lat-long into clusters of regions and then use the region varaible in the model, but I believe we are not focusing on model explainability in this exercise.
airbnbDF2 = airbnbDF.drop(columns=['zipcode']).copy()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the string columns that you've decided to keep, pick a numerical encoding for the string columns. Don't forget to deal with the `NaN` entries in those columns first.

# COMMAND ----------

import numpy as np

# superhost_binary_code = {"t": 1, "f": 0}
airbnbDF2['host_is_superhost_'] = np.where(airbnbDF2['host_is_superhost'] == 't', 1, 0)

cancellation_policy_num_scale = {"flexible": 0, "moderate": 1, "strict": 2, "super_strict_30": 3, "super_strict_60": 4}
airbnbDF2['cancellation_policy_strictness'] = airbnbDF2['cancellation_policy'].replace(cancellation_policy_num_scale)

# instant_bookable_binary_code = {"t": 1, "f": 0}
airbnbDF2['instant_bookable_'] = np.where(airbnbDF2['instant_bookable'] == 't', 1, 0)

# One hot encode the following variables: 'neighbourhood_cleansed', 'property_type', 'bed_type', 'room_type'
airbnbDF2 = pd.get_dummies(airbnbDF2, columns=["neighbourhood_cleansed", "property_type", "bed_type", "room_type"], prefix=["neighbourhood", "property", "bed", "room"], dtype=np.int64)

airbnbDF2 = airbnbDF2.drop(columns=['cancellation_policy', 'host_is_superhost', 'instant_bookable'])

display(airbnbDF2)

# COMMAND ----------

# Check the data types of columns available in the final data
print("Available data types are:")
print(airbnbDF2.dtypes.value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC Before we create a train test split, check that all your columns are numerical. Remember to drop the original string columns after creating numerical representations of them.
# MAGIC 
# MAGIC Make sure to drop the price column from the training data when doing the train test split.

# COMMAND ----------

# TODO
from sklearn.model_selection import train_test_split

target_var = 'price'
input_vars = airbnbDF2.columns.tolist()
input_vars.remove('price')

X, y = airbnbDF2[input_vars], airbnbDF2[target_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model
# MAGIC 
# MAGIC After cleaning our data, we can start creating our model!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Firstly, if there are still `NaN`'s in your data, you may want to impute these values instead of dropping those entries entirely. Make sure that any further processing/imputing steps after the train test split is part of a model/pipeline that can be saved.
# MAGIC 
# MAGIC In the following cell, create and fit a single sklearn model.

# COMMAND ----------

# Print the names of columns with at least one missing value 
airbnbDF2.columns[airbnbDF2.isnull().any()]

# COMMAND ----------

# TODO
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Missing value imputation of numeric varaibles
missing_vars_1 = ['host_total_listings_count', 'bathrooms', 'beds']
missing_imputation_1 = SimpleImputer(strategy="median")

# The review related variables have missing values for those records where number of reviews is 0
#     - One way to go ahead is to build two models (1. hotels with reviews & 2. hotels without reviews) (I think it's out of scope of this exercise)
#     - We impute with -99 (a distant negative value). This is okay because we will use tree based ML techniques for regression.
missing_vars_2 = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']
missing_imputation_2 = SimpleImputer(strategy="constant", fill_value=-99)

preprocessor = ColumnTransformer(
    transformers=[
        ("mvi1", missing_imputation_1, missing_vars_1),
        ("mvi2", missing_imputation_2, missing_vars_2),
    ]
)

params = {'n_estimators': 100 , 
          'max_depth': 25 ,
          'bootstrap': True ,
          'max_features': 'auto' ,
          'random_state': 42
         }

clf = Pipeline(
    steps=[("Preprocessor", preprocessor), ("RF", RandomForestRegressor(**params))]
)

clf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Pick and calculate a regression metric for evaluating your model.

# COMMAND ----------

# TODO
from sklearn.metrics import mean_squared_error

predictions = clf.predict(X_test)    

mse = mean_squared_error(y_test, predictions)
print(f"mse: {mse}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Log your model on MLflow with the same metric you calculated above so we can compare all the different models you have tried! Make sure to also log any hyperparameters that you plan on tuning!

# COMMAND ----------

# TODO
import mlflow.sklearn

with mlflow.start_run(run_name = "First Run") as run:
    
    # Log model with name
    mlflow.sklearn.log_model(clf, "mvi_and_rf_model__pipeline")
    
    # Log params
    all_params = clf.get_params()['RF'].get_params()
    params_ = {x: all_params[x] for x in params}
    mlflow.log_params(params_)
    
    # Create and log MSE metrics using predictions of X_test and its actual value y_test
    mlflow.log_metric("mse", mse)
    
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Change and re-run the above 3 code cells to log different models and/or models with different hyperparameters until you are satisfied with the performance of at least 1 of them.

# COMMAND ----------

# Instead of editing and running the above cells, we will define a function to do grid serach of hyper parameters, and log the best model
def log_best_model_grid_search(experimentID, run_name, params_, X_train, X_test, y_train, y_test):
    
    from sklearn.model_selection import GridSearchCV

    with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
        
        clf_ = Pipeline(
            steps=[("Preprocessor", preprocessor), ("RF", RandomForestRegressor())]
        )

        # Modify the parameter names so that they apply only to the 'Random Forest' step in the pipeline
        parameters_grid = {'RF__'+str(key): item for key, item in params_.items()}
        
        grid_clf_ = GridSearchCV(clf_, parameters_grid, cv=3)
        grid_clf_.fit(X_train, y_train)

        best_clf_ = grid_clf_.best_estimator_
    
        # Create model, train it, and create predictions
        best_clf_.fit(X_train, y_train)
        predictions = best_clf_.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Log model
        mlflow.sklearn.log_model(best_clf_, "mvi_and_rf_model__pipeline")

        # Log params
        all_params = best_clf_.get_params()['RF'].get_params()
        params__ = {x: all_params[x] for x in params_}
        print("Best parameters are:")
        print(params__)
        mlflow.log_params(params__)

        # Log metrics
        mlflow.log_metric("mse", mse)
        print(f"mse: {mse}")
        
        return run.info.run_uuid

# COMMAND ----------

# Grid Serach 1 - A very wide search to narrow down to a good spot of hyperparameter values
parameters = {'n_estimators': [int(x) for x in range(10, 311, 30)] , 
              'max_depth': [int(x) for x in range(10, 101, 10)] ,
              'bootstrap': [True, False] ,
              'max_features': ['auto', 'sqrt'] ,
              'random_state': [42]
             }
log_best_model_grid_search(experimentID, "Grid Search - Run 1", parameters, X_train, X_test, y_train, y_test)

# COMMAND ----------

# Grid Serach 2 - A narrower search in the hyperparameter area identified in gird search 1
parameters = {'n_estimators': [int(x) for x in range(100, 201, 10)] , 
              'max_depth': [int(x) for x in range(1, 22, 2)] ,
              'bootstrap': [True] ,
              'max_features': ['auto', 'sqrt'] ,
              'random_state': [42]
             }
log_best_model_grid_search(experimentID, "Grid Search - Run 2", parameters, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Look through the MLflow UI for the best model. Copy its `URI` so you can load it as a `pyfunc` model.

# COMMAND ----------

# TODO
import mlflow.pyfunc

model_uri = "runs:/bc98aaf8eed645118ca9a0f67f078c35/mvi_and_rf_model__pipeline"

model_version_1 = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-processing
# MAGIC 
# MAGIC Our model currently gives us the predicted price per night for each Airbnb listing. Now we would like our model to tell us what the price per person would be for each listing, assuming the number of renters is equal to the `accommodates` value. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Fill in the following model class to add in a post-processing step which will get us from total price per night to **price per person per night**.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Check out <a href="https://www.mlflow.org/docs/latest/models.html#id13" target="_blank">the MLFlow docs for help.</a>

# COMMAND ----------

# TODO

class Airbnb_Model(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        model_ = self.model
        model_out = model_.predict(model_input)/model_input['accommodates']
        return model_out


# COMMAND ----------

# MAGIC %md
# MAGIC Construct and save the model to the given `final_model_path`.

# COMMAND ----------

# TODO
from mlflow.exceptions import MlflowException

Airbnb_Model_v1 = Airbnb_Model(model=model_version_1)

final_model_path =  f"{working_path}/final-model"
dbutils.fs.rm(final_model_path, True) # Allows you to rerun the code multiple times
mlflow.pyfunc.save_model(path=final_model_path.replace("dbfs:", "/dbfs"), python_model=Airbnb_Model_v1)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the model in `python_function` format and apply it to our test data `X_test` to check that we are getting price per person predictions now.

# COMMAND ----------

# TODO
loaded_model = mlflow.pyfunc.load_model(final_model_path)

model_input = X_test
model_output = loaded_model.predict(model_input)

assert model_output.equals(model_version_1.predict(X_test)/X_test['accommodates'])
print("Success - Predictions are price per person")

model_output

# COMMAND ----------

# MAGIC %md
# MAGIC ## Packaging your Model
# MAGIC 
# MAGIC Now we would like to package our completed model! 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC First save your testing data at `test_data_path` so we can test the packaged model.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** When using `.to_csv` make sure to set `index=False` so you don't end up with an extra index column in your saved dataframe.

# COMMAND ----------

# TODO
# save the testing data 
test_data_path = f"{working_path}/test_data.csv"
X_test.to_csv(test_data_path, index=False)

prediction_path = f"{working_path}/predictions.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC First we will determine what the project script should do. Fill out the `model_predict` function to load out the trained model you just saved (at `final_model_path`) and make price per person predictions on the data at `test_data_path`. Then those predictions should be saved under `prediction_path` for the user to access later.
# MAGIC 
# MAGIC Run the cell to check that your function is behaving correctly and that you have predictions saved at `demo_prediction_path`.

# COMMAND ----------

# TODO
import click
import mlflow.pyfunc
import pandas as pd

@click.command()
@click.option("--final_model_path", default="", type=str)
@click.option("--test_data_path", default="", type=str)
@click.option("--prediction_path", default="", type=str)

def model_predict(final_model_path, test_data_path, prediction_path):
    
    loaded_model = mlflow.pyfunc.load_model(final_model_path)
    model_input = pd.read_csv(test_data_path)
    predictions = loaded_model.predict(model_input)
    predictions.to_csv(prediction_path, index=False)

# test model_predict function    
demo_prediction_path = f"{working_path}/predictions.csv"

from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(model_predict, ['--final_model_path', final_model_path, 
                                       '--test_data_path', test_data_path,
                                       '--prediction_path', demo_prediction_path], catch_exceptions=True)

assert result.exit_code == 0, "Code failed" # Check to see that it worked
print("Price per person predictions: ")
print(pd.read_csv(demo_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will create a MLproject file and put it under our `workingDir`. Complete the parameters and command of the file.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/MLproject", 
'''
name: Capstone-Project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      final_model_path: {type: str, default: "/dbfs/user/pthulasi@ur.rochester.edu/mlflow/99_putting_it_all_together_psp/final-model"}
      test_data_path: {type: str, default: "/dbfs/user/pthulasi@ur.rochester.edu/mlflow/99_putting_it_all_together_psp/test_data.csv"}
      prediction_path: {type: str, default: "/dbfs/user/pthulasi@ur.rochester.edu/mlflow/99_putting_it_all_together_psp/predictions.csv"}
    command: "python predict.py --final_model_path {final_model_path} --test_data_path {test_data_path} --prediction_path {prediction_path}"
'''.strip(), overwrite=True)

# COMMAND ----------

print(prediction_path)

# COMMAND ----------

# MAGIC %md
# MAGIC We then create a `conda.yaml` file to list the dependencies needed to run our script.
# MAGIC 
# MAGIC For simplicity, we will ensure we use the same version as we are running in this notebook.

# COMMAND ----------

import cloudpickle, numpy, pandas, sklearn, sys

version = sys.version_info # Handles possibly conflicting Python versions

file_contents = f"""
name: Capstone
channels:
  - defaults
dependencies:
  - python={version.major}.{version.minor}.{version.micro}
  - cloudpickle={cloudpickle.__version__}
  - numpy={numpy.__version__}
  - pandas={pandas.__version__}
  - scikit-learn={sklearn.__version__}
  - pip:
    - mlflow=={mlflow.__version__}
""".strip()

dbutils.fs.put(f"{workingDir}/conda.yaml", file_contents, overwrite=True)

print(file_contents)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will put the **`predict.py`** script into our project package.
# MAGIC 
# MAGIC Complete the **`.py`** file by copying and placing the **`model_predict`** function you defined above.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/predict.py", 
'''
import click
import mlflow.pyfunc
import pandas as pd

# TODO
import click
import mlflow.pyfunc
import pandas as pd

@click.command()
@click.option("--final_model_path", default="", type=str)
@click.option("--test_data_path", default="", type=str)
@click.option("--prediction_path", default="", type=str)

def model_predict(final_model_path, test_data_path, prediction_path):
    
    loaded_model = mlflow.pyfunc.load_model(final_model_path)
    model_input = pd.read_csv(test_data_path)
    predictions = loaded_model.predict(model_input)
    predictions.to_csv(prediction_path, index=False)
    
if __name__ == "__main__":
    model_predict()
'''.strip(), overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's double check all the files we've created are in the `workingDir` folder. You should have at least the following 3 files:
# MAGIC * `MLproject`
# MAGIC * `conda.yaml`
# MAGIC * `predict.py`

# COMMAND ----------

display( dbutils.fs.ls(workingDir) )

# COMMAND ----------

# MAGIC %md
# MAGIC Under **`workingDir`** is your completely packaged project.
# MAGIC 
# MAGIC Run the project to use the model saved at **`final_model_path`** to predict the price per person of each Airbnb listing in **`test_data_path`** and save those predictions under **`second_prediction_path`** (defined below).

# COMMAND ----------

# TODO
second_prediction_path = f"{working_path}/predictions-2.csv"
mlflow.projects.run(working_path,
  parameters={
    "final_model_path": final_model_path,
    "test_data_path": test_data_path,
    "prediction_path": second_prediction_path
})

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to check that your model's predictions are there!

# COMMAND ----------

print("Price per person predictions: ")
print(pd.read_csv(second_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> All done!</h2>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
