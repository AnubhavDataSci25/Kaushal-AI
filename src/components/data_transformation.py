import os
import sys
from dataclasses import dataclass

import ast
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.execption import CustomException
from src.logger.loggings import logging
from src.utils import save_objects

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    target_encoder_obj_file_path = os.path.join("artifacts", "target_encoder.pkl")

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        X_list = self._extract(X)
        self.mlb.fit(X_list)
        return self

    def transform(self, X):
        X_list = self._extract(X)
        return self.mlb.transform(X_list)

    def _extract(self, X):
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            if X.ndim == 2 and X.shape[1] == 1:
                series = X[:, 0]
            else:
                series = X.ravel()
        else:
            series = X

        out = []
        for elem in series:
            if isinstance(elem, (list, tuple, np.ndarray)):
                out.append(list(elem))
            elif pd.isna(elem):
                out.append([])
            elif isinstance(elem, str):
                items = [s.strip() for s in elem.strip("[]\"' ").split(",") if s.strip()]
                out.append(items)
            else:
                out.append([elem])
        return out


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _parse_interests(self, value):
        if pd.isna(value):
            return value
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    if len(parsed) == 0:
                        return np.nan
                    # if list has multiple interests, join with comma to keep a single category feature
                    return ", ".join(str(x).strip() for x in parsed)
            except Exception:
                pass
            return value.strip("[]\"' ")
        return value

    def _parse_skills(self, value):
        if pd.isna(value):
            return []
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed]
            except Exception:
                # fallback: split by commas
                parts = [item.strip() for item in value.strip("[]").split(",") if item.strip()]
                return parts
            return [value.strip()]
        if isinstance(value, (list, tuple)):
            return [str(x).strip() for x in value]
        return [str(value)]

    def get_data_transformer_obj(self, available_columns=None):
        try:
            num_features = ['experience_years', 'projects_count']
            cat_features_ohe = ['education', 'interests', 'certification', 'learning_source', 'dominant_project_domain']
            skills_feature_mlb = ['skills']

            if available_columns is not None:
                available_columns = set(available_columns)
                num_features = [col for col in num_features if col in available_columns]
                cat_features_ohe = [col for col in cat_features_ohe if col in available_columns]
                skills_feature_mlb = [col for col in skills_feature_mlb if col in available_columns]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
            ])

            cat_pipeline_ohe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="constant", fill_value="NA")),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')),
            ])

            skills_pipeline_mlb = MultiLabelBinarizerTransformer()

            # Only include the intended numeric and categorical feature pipelines.
            # Drop any unspecified columns (e.g. id fields) so string values
            # don't get passthrough into the numeric model inputs.
            transformers = []

            if num_features:
                transformers.append(('num', num_pipeline, num_features))
            if cat_features_ohe:
                transformers.append(('cat_ohe', cat_pipeline_ohe, cat_features_ohe))
            if skills_feature_mlb:
                transformers.append(('skills_mlb', skills_pipeline_mlb, skills_feature_mlb))

            preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation Started")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data read successfully")

            # Remove user_id column if exists as it is not a useful feature and can cause issues in transformation
            train_df = train_df.drop(columns=['user_id'], axis=1, errors='ignore')
            test_df = test_df.drop(columns=['user_id'], axis=1, errors='ignore')

            # Normalize list-like fields before transformer.
            for df in [train_df, test_df]:
                if 'skills' in df.columns:
                    df['skills'] = df['skills'].apply(self._parse_skills)
                if 'interests' in df.columns:
                    df['interests'] = df['interests'].apply(self._parse_interests)

            logging.info("Obtaining preprocessing Object")
            preprocessing_obj = self.get_data_transformer_obj(input_features_train.columns)

            target_col_name = 'target_career'

            input_features_train = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]
            
            input_features_test = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessing_obj.transform(input_features_test)

            target_encoder = LabelEncoder()
            target_feature_train_arr = target_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = target_encoder.transform(target_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_features_test_arr, target_feature_test_arr]
            logging.info("Saved preprocessing object and target encoder object")

            save_objects(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            save_objects(
                file_path=self.data_transformation_config.target_encoder_obj_file_path,
                obj=target_encoder
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.target_encoder_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
