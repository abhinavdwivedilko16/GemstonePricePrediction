import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')  #this code is written so that it can be worked in both linux and windows after deployment
            model_path= os.path.join('artifacts','model.pkl')

            #to load the pickle file, code has been written in utils.py
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Error occoured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df
        except Exception as e:
            logging.info("Exception occoured in prediction pipeline")
            raise CustomException(e,sys)
        