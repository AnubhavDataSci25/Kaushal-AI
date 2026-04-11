import pandas as pd
import os, sys
from src.execption import CustomException
from src.utils import load_objects

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_objects(file_path=model_path)
            preprocessor = load_objects(file_path=preprocessor_path)
            target_encoder_path = os.path.join('artifacts', 'target_encoder.pkl')
            target_encoder = load_objects(file_path=target_encoder_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            pred = target_encoder.inverse_transform(pred)
            return pred
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(
        self,
        education: str,
        experience_years: int,
        skills: list,
        interests: str,
        certifications: str
        ):

        self.education = education
        self.experience_years = experience_years
        self.skills = skills
        self.interests = interests
        self.certifications = certifications

    def get_data_as_df(self):
        try:
            data_dict = {
                'education': [self.education],
                'experience_years': [self.experience_years],
                'skills': [', '.join(self.skills)],
                'interests': [self.interests],
                'certifications': [self.certifications]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)