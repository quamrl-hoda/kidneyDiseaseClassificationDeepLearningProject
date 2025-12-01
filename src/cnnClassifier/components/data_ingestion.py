import os
import zipfile
import gdown
from cnnClassifier import logger
from pathlib import Path
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
  def __init__(self, config: DataIngestionConfig):
    self.config = config

  def download_file(self,) -> str:
    """Fetch data from google drive
    """
    try:
      dataset_url = self.config.source_url
      zip_download_dir = self.config.local_data_file
      os.makedirs("artifacts/data_ingestion", exist_ok=True)
      logger.info(f"downloading data from {dataset_url} to {zip_download_dir}")

      file_id = dataset_url.split("/")[-2]
      prefix = "https://drive.google.com/uc?export=download&id="
      gdown.download(prefix+file_id, zip_download_dir)

      logger.info(f"Downloaded data from {dataset_url} to {zip_download_dir}")

    except Exception as e:
      raise e
  
  def extract_zip_file(self,)-> None:
    """Extract zip file to directory
    """
    unzip_path = self.config.unzip_dir
    os.makedirs(unzip_path, exist_ok=True)
    with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
      logger.info(f"Extracting zip file: {self.config.local_data_file} to dir: {unzip_path}")
      zip_ref.extractall(unzip_path)
      logger.info(f"Extracted zip file: {self.config.local_data_file} to dir: {unzip_path} and the size of data is {get_size(Path(unzip_path))}")
