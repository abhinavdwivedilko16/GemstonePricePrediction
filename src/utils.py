import os 
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from src.exception import CustomException
from src.logger import logging
import pickle

def save_object(file_path,obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report= {}
        for i in range(len(models)):
            model=list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)


            #predict testing data
            y_test_pred = model.predict(X_test)

            #get r2 score for the train and test data
            ### train_model_score= r2_score(y_train,y_test_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

            return report
    except Exception as e:
        logging.info('exception occoured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Exception occoured in load object function in utils")
        raise CustomException(e,sys)
    