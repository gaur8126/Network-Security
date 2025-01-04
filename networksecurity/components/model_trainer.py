
import os 
import sys 
from networksecurity.logger.logger import logging 
from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from xgboost import XGBClassifier
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import load_numpy_array_data,save_object,load_object
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.model_selection import GridSearchCV
# import optuna
# from optuna.integration import XGBoostPruningCallback

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def perform_hyper_parameter_tuning(self,clf,x_train,y_train):
        pass
    
        # params = {
        #     "booster":['gbtree','gblinear','dart'],
        #     "verbosity" : [0,1,2,3],
        #     "eta":[0.1,0.2,0.3,0.4,0.5],
        #     "gamma":[0,1,2,3,4,5],
        #     "max_depth":[3,5,6,8,9]
        # }
        # logging.info("Tuning has started")
        # gcv  = GridSearchCV(clf,cv=5,verbose=3,param_grid=params)
        # gcv.fit(x_train,y_train)
        # best_params = gcv.best_params_
        # logging.info(f"best parameters : {best_params}")
        # return best_params


        # def objective(trial):
        #     param = {
        #         "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        #         "verbosity": trial.suggest_int("verbosity", 0, 3),
        #         "eta": trial.suggest_float("eta", 0.1, 0.5),
        #         "gamma": trial.suggest_int("gamma", 0, 5),
        #         "max_depth": trial.suggest_int("max_depth", 3, 9),
        #     }
        #     model = XGBClassifier(**param)
        #     model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=10)
        #     return model.score(x_test, y_test)

        # study = optuna.create_study(direction="maximize")
        # study.optimize(objective, n_trials=50)
        # print("Best parameters:", study.best_params)


    def train_model(self,x_train,y_train):
        try:
            logging.info("Training has started")
            # best_params = self.perform_hyper_parameter_tuning(clf=XGBClassifier(),x_train=x_train,y_train = y_train)
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train,y_train)
            logging.info("Model training has completed")
            return xgb_clf
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Training file path")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading trainer array and testing array 
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],

            )

            model = self.train_model(x_train,y_train)
            y_train_pred = model.predict(x_train)

            classification_train_metric = get_classification_score(y_true = y_train,y_pred=y_train_pred)

            if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuaracy:
                print("Trained model is not good to provide expected accurcy")

            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)

            # Overfitting and Underfitting 

            diff = abs(classification_train_metric.f1_score-classification_test_metric.f1_score)

            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")
            
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            Network_Model = NetworkModel(preprocessor=preprocessor,model=model)
            save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)

            # model trainer artifact 

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact =  classification_train_metric,
                test_metric_artifact = classification_test_metric)
            logging.info(f"Model trainer artifact : {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e :
            raise NetworkSecurityException(e,sys)