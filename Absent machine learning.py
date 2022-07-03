

#A classification pipeline which aim at predicting the amount of hours
# a worker will be absent from work based on the worker characteristics and the work day missed.


import numpy as np
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def load_dataset(train_csv_path):
    data = pd.read_csv(train_csv_path, sep=',')
    return data


class DataPreprocessor(object):



    def __init__(self):
      self.transformer:Pipeline = None

    def fit(self, dataset_df):

        numerical_columns = ['Transportation expense', 'Height',] 
        # numerical_columns = ['Reason', 'Education']
        # numerical_columns = ['Transportation expense', 'Residence Distance', 'Service time', 'Weight', 'Height',]
        categorical_columns = list(set(dataset_df.columns) - set(numerical_columns))


        # Handling Numerical Fields
        num_pipeline = Pipeline([
          ('imputer', SimpleImputer(strategy="mean"))
        ])


        # Handling Categorical Fields
        categorical_transformer = OneHotEncoder(drop=None, sparse=False, handle_unknown='ignore')
        cat_pipeline = Pipeline([
          ('imputer', SimpleImputer(strategy="most_frequent")),('1hot', categorical_transformer)
        ])

        # preprocessor
        preprocessor = ColumnTransformer(
          transformers=[
            ("num", num_pipeline, numerical_columns),
            ("cat", cat_pipeline, categorical_columns),
          ]
        )
      
        self.transformer = Pipeline(steps=[
          ("preprocessor", preprocessor)
        ])

        self.transformer.fit(dataset_df)


    def transform(self, df):

        df['bmi'] = df['Weight']/((df['Height']/100)**2)

        return self.transformer.transform(df)
       



def train_model(processed_X, y):
  

 
    model = RandomForestClassifier()


    model.fit(processed_X, y)

    return model
    
    
    
preprocessor = DataPreprocessor()
train_csv_path = 'time_off_data_train.csv'
train_dataset_df = load_dataset(train_csv_path)
sum_test_score = 0
sum_train_score = 0
for i in [1,2,3,4,5,6,7,8,9,10]:


  start = (i-1)*50
  end = i*50
  train_dataset_df_cv = train_dataset_df.drop(train_dataset_df.index[start:end])
  validation_train_dataset_df = train_dataset_df.iloc[start:end, :]
  

  X_train = train_dataset_df_cv.iloc[:, :-1]
  y_train = train_dataset_df_cv['TimeOff']
  preprocessor.fit(X_train)
  model = train_model(preprocessor.transform(X_train), y_train)



  X_test = validation_train_dataset_df.iloc[:, :-1]
  y_test = validation_train_dataset_df['TimeOff']

  processed_X_test = preprocessor.transform(X_test)
  predictions = model.predict(processed_X_test)
  test_score = accuracy_score(y_test, predictions)
  print("test:", test_score)

  sum_test_score += test_score

  predictions = model.predict(preprocessor.transform(X_train))
  test_score = accuracy_score(y_train, predictions)
  print('train:', test_score)
  print('\n')
  sum_train_score += test_score



print(sum_test_score/10)
print(sum_train_score/10)        
