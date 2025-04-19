import os
import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def train_model(self):
        try:
        
            data_path = os.path.join("artifacts", "data.csv")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure it has been created.")

            df = pd.read_csv(data_path)

            X = df.drop(columns=["math_score"]) 
            y = df["math_score"]

      
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            
            os.makedirs(os.path.dirname(self.preprocessor_path), exist_ok=True)
            with open(self.preprocessor_path, "wb") as f:
                pickle.dump(scaler, f)

            logging.info("Preprocessing complete. Preprocessor saved.")

            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

           
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)

            logging.info("Model training complete. Model saved.")

            return self.model_path, self.preprocessor_path
        
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    trainer = TrainPipeline()
    trainer.train_model()
