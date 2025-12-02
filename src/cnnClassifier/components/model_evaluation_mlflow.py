import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import os
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json

class Evaluation:
  def __init__(self, config:EvaluationConfig):
    self.config = config

  def _valid_generator(self):

    datagenerator_kwargs = dict(
      rescale=1./255,
      validation_split=0.30 # 30% data for validation
    )
    dataflow_kwargs = dict(
      target_size=self.config.params_image_size[:-1],
      batch_size=self.config.params_batch_size,
      class_mode="categorical"
    )

    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

    self.valid_generator = valid_datagenerator.flow_from_directory(
      directory=self.config.training_data,
      subset="validation",
      shuffle=False,  
      **dataflow_kwargs
    )

  @staticmethod
  def load_model(path:Path) -> tf.keras.Model:
      model = tf.keras.models.load_model(path)
      return model
    
  def evaluation(self):
      self.model = self.load_model(self.config.path_of_model)
      self._valid_generator()
      self.score = self.model.evaluate(self.valid_generator)
      self.save_score()
  # def save_score(self):
  #     scores = {"loass":self.score[0], "accuracy":self.score[1]}
  #     save_json(Path("artifacts/evaluation/scores.json"), data=scores)
  def save_score(self):
    scores = {"loss": self.score[0], "accuracy": self.score[1]}
    # create folder if not exists
    os.makedirs("artifacts/evaluation", exist_ok=True)
    save_json(Path("artifacts/evaluation/scores.json"), data=scores)

  def log_into_mlflow(self):

    # Clean parameters for MLflow
    clean_params = {}
    for key, value in self.config.all_params.items():
        if isinstance(value, (list, tuple, dict)):
            clean_params[key] = str(value)
        else:
            clean_params[key] = value

    mlflow.log_params(clean_params)

    # Log metrics
    mlflow.log_metrics({
        "eval_loss": self.score[0],
        "eval_accuracy": self.score[1]
    })

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.keras.log_model(
            self.model,
            "model",
            registered_model_name="VGG16Model"
        )
    else:
        mlflow.keras.log_model(self.model, "model")
