"""
Optimized Training Pipeline - Top 15 Features, 3 Models, 72h Prediction
"""

import pandas as pd
import numpy as np
import hopsworks
import os
from dotenv import load_dotenv
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


try:
    import tensorflow as tf
    from tensorflow import keras
    # Use attributes from the imported keras object instead of importing tensorflow.keras directly
    layers = keras.layers
    callbacks = keras.callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from feature_pipeline import AQIFeatureEngineer

load_dotenv()


class AQIModelTrainer:
    
    def __init__(self):
        self.hopsworks_api_key = os.getenv('HOPSWORKS_API_KEY')
        self.project_name = os.getenv('HOPSWORKS_PROJECT_NAME', 'ubaidrazaaqi')
        
        if not self.hopsworks_api_key:
            raise ValueError("HOPSWORKS_API_KEY not found in .env file")
        
        self.feature_engineer = AQIFeatureEngineer()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.top_features = None
        
        print(f"✓ Initialized training pipeline")
    
    def connect_to_hopsworks(self):
        print(f"\nConnecting to Hopsworks...")
        project = hopsworks.login(api_key_value=self.hopsworks_api_key, project=self.project_name)
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        print("✓ Connected to Hopsworks")
        return project, fs, mr
    
    def fetch_features(self, fs, feature_group_name="aqi_features", version=1):
        print(f"\nFetching features from {feature_group_name}_v{version}")
        fg = fs.get_feature_group(name=feature_group_name, version=version)
        df = fg.read()
        print(f"✓ Fetched {len(df)} records")
        return df
    
    def select_top_features(self, X_train, y_train, n_features=15):
        """Select top N features using Random Forest importance"""
        print(f"\nSelecting top {n_features} features...")
        
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train, y_train)
        
        importances = rf_temp.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.top_features = feature_importance.head(n_features)['feature'].tolist()
        
        print(f"✓ Top {n_features} features selected:")
        for i, (idx, row) in enumerate(feature_importance.head(n_features).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:30s} : {row['importance']:.4f}")
        
        return self.top_features
    
    def prepare_data(self, df, target_column='target_aqi_72h', test_size=0.2):
        """Prepare train/test data with feature selection"""
        print(f"\nPreparing data for target: {target_column}")
        
        df_clean = df.dropna(subset=[target_column]).copy()
        
        exclude_cols = ['date', 'target_aqi_1h', 'target_aqi_6h', 'target_aqi_12h', 
                       'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean[target_column]
        X = X.fillna(X.mean())
        
        split_idx = int(len(df_clean) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Select top features
        self.select_top_features(X_train, y_train, n_features=15)
        
        # Use only top features
        X_train = X_train[self.top_features]
        X_test = X_test[self.top_features]
        
        print(f"\n✓ Data prepared:")
        print(f"  Training: {len(X_train)}, Testing: {len(X_test)}")
        print(f"  Features: {len(self.top_features)}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        print("\n[1/3] Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.models['RandomForest'] = rf
        self.results['RandomForest'] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        return rf
    
    def train_ridge(self, X_train, y_train, X_test, y_test):
        print("\n[2/3] Training Ridge Regression...")
        ridge = Ridge(alpha=10.0, random_state=42)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.models['Ridge'] = ridge
        self.results['Ridge'] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        return ridge
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        if not TENSORFLOW_AVAILABLE:
            print("\n[3/3] Skipping Neural Network (TensorFlow not available)")
            return None
        
        print("\n[3/3] Training Neural Network...")
        
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
        
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, 
                  callbacks=[early_stop], verbose=0)
        
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.models['NeuralNetwork'] = model
        self.results['NeuralNetwork'] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        return model
    
    def select_best_model(self):
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        results_df = pd.DataFrame(self.results).T.sort_values('RMSE')
        print(results_df)
        
        best_name = results_df.index[0]
        best_metrics = results_df.iloc[0]
        best_model = self.models[best_name]
        
        print(f"\n✓ Best Model: {best_name}")
        print(f"  RMSE: {best_metrics['RMSE']:.4f}")
        print(f"  MAE: {best_metrics['MAE']:.4f}")
        print(f"  R²: {best_metrics['R2']:.4f}")
        
        return best_model, best_name, best_metrics
    
    def save_model(self, mr, model, model_name, metrics, target_column):
        print(f"\nSaving model...")
        
        model_dir = "aqi_model_72h"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        if isinstance(model, keras.Model):
            model.save(os.path.join(model_dir, "model.h5"))
        else:
            joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        
        # Save scaler and metadata
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
        
        metadata = {
            'model_type': model_name,
            'target_column': target_column,
            'feature_names': self.top_features,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'training_date': datetime.now().isoformat(),
            'n_features': len(self.top_features)
        }
        
        with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Register in Hopsworks
        try:
            aqi_model = mr.python.create_model(
                name="aqi_prediction_72h",
                metrics={k: float(v) for k, v in metrics.items()},
                description=f"AQI 72h prediction using {model_name} with top 15 features"
            )
            aqi_model.save(model_dir)
            print(f"✓ Model saved to registry")
        except Exception as e:
            print(f"⚠ Registry save failed: {e}")
            print(f"✓ Model saved locally to {model_dir}")
    
    def run_training(self, feature_group_name="aqi_features", version=1):
        print("="*60)
        print("AQI TRAINING PIPELINE - 24H, 48H, 72H PREDICTIONS")
        print("="*60)
        
        project, fs, mr = self.connect_to_hopsworks()
        df = self.fetch_features(fs, feature_group_name, version)
        
        # Train models for each horizon
        for target, hours in [('target_aqi_24h', 24), ('target_aqi_48h', 48), ('target_aqi_72h', 72)]:
            print(f"\n{'='*60}")
            print(f"TRAINING FOR {hours}H PREDICTION")
            print(f"{'='*60}")
            
            X_train, X_test, y_train, y_test = self.prepare_data(df, target)
            
            print("\nTraining models...")
            self.train_random_forest(X_train, y_train, X_test, y_test)
            self.train_ridge(X_train, y_train, X_test, y_test)
            self.train_neural_network(X_train, y_train, X_test, y_test)
            
            best_model, best_name, best_metrics = self.select_best_model()
            
            # Save with horizon-specific name
            model_dir = f"aqi_model_{hours}h"
            self.save_model_to_dir(model_dir, best_model, best_name, best_metrics, target)
            
            # Clear models for next iteration
            self.models = {}
            self.results = {}
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINING COMPLETED!")
        print("="*60)
        print("Models saved:")
        print("  • aqi_model_24h/")
        print("  • aqi_model_48h/")
        print("  • aqi_model_72h/")
    
    def save_model_to_dir(self, model_dir, model, model_name, metrics, target_column):
        """Save model to specific directory"""
        print(f"\nSaving model to {model_dir}...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        if isinstance(model, keras.Model):
            model.save(os.path.join(model_dir, "model.h5"))
        else:
            joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        
        # Save scaler and metadata
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
        
        metadata = {
            'model_type': model_name,
            'target_column': target_column,
            'feature_names': self.top_features,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'training_date': datetime.now().isoformat(),
            'n_features': len(self.top_features)
        }
        
        with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to {model_dir}")


if __name__ == "__main__":
    try:
        trainer = AQIModelTrainer()
        trainer.run_training(feature_group_name="aqi_features", version=1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()