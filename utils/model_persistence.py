"""
Model Persistence Module
Handles saving and loading trained ML models with metadata.
"""

import os
import json
import joblib
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelPersistence:
    """
    Manages saving and loading trained ML models with all necessary artifacts.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize ModelPersistence.

        Args:
            base_dir: Base directory for model storage (defaults to project_root/models)
        """
        if base_dir is None:
            # Get project root (2 levels up from utils/)
            project_root = Path(__file__).parent.parent
            base_dir = project_root / "models"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model storage directory: {self.base_dir}")

    def save_model(
        self,
        model: Any,
        preprocessor: Any,
        metadata: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Save trained model with all artifacts.

        Args:
            model: Trained scikit-learn compatible model
            preprocessor: Fitted preprocessing pipeline (scaler, encoders, etc.)
            metadata: Model metadata (features, metrics, task_type, etc.)
            model_id: Optional custom model ID (generates UUID if not provided)

        Returns:
            Tuple of (model_id, model_directory_path)
        """
        try:
            # Generate model ID if not provided
            if model_id is None:
                model_id = str(uuid.uuid4())

            # Create model directory
            model_dir = self.base_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"💾 Saving model to: {model_dir}")

            # Save model
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"✅ Model saved: {model_path}")

            # Save preprocessor
            preprocessor_path = model_dir / "preprocessor.pkl"
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"✅ Preprocessor saved: {preprocessor_path}")

            # Add timestamp and model_id to metadata
            metadata["model_id"] = model_id
            metadata["saved_at"] = datetime.now().isoformat()
            metadata["model_path"] = str(model_path)
            metadata["preprocessor_path"] = str(preprocessor_path)

            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✅ Metadata saved: {metadata_path}")

            logger.info(f"🎉 Model package saved successfully: {model_id}")

            return model_id, str(model_dir)

        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise e

    def load_model(self, model_id: str) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load trained model with all artifacts.

        Args:
            model_id: Model identifier (UUID)

        Returns:
            Tuple of (model, preprocessor, metadata)

        Raises:
            FileNotFoundError: If model directory or files don't exist
        """
        try:
            model_dir = self.base_dir / model_id

            if not model_dir.exists():
                raise FileNotFoundError(f"Model not found: {model_id}")

            logger.info(f"📂 Loading model from: {model_dir}")

            # Load model
            model_path = model_dir / "model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = joblib.load(model_path)
            logger.info(f"✅ Model loaded")

            # Load preprocessor
            preprocessor_path = model_dir / "preprocessor.pkl"
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"✅ Preprocessor loaded")

            # Load metadata
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"✅ Metadata loaded")

            logger.info(f"🎉 Model package loaded successfully: {model_id}")

            return model, preprocessor, metadata

        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {e}")
            raise e

    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get model metadata without loading the full model.

        Args:
            model_id: Model identifier

        Returns:
            Model metadata dictionary
        """
        try:
            model_dir = self.base_dir / model_id
            metadata_path = model_dir / "metadata.json"

            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found for model: {model_id}")

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return metadata

        except Exception as e:
            logger.error(f"❌ Failed to load metadata for {model_id}: {e}")
            raise e

    def list_models(self) -> list:
        """
        List all available models.

        Returns:
            List of model metadata dictionaries
        """
        try:
            models = []

            # Iterate through all subdirectories
            for model_dir in self.base_dir.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            models.append(metadata)

            # Sort by save date (newest first)
            models.sort(key=lambda x: x.get("saved_at", ""), reverse=True)

            logger.info(f"📋 Found {len(models)} saved models")

            return models

        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            return []

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a saved model and all its artifacts.

        Args:
            model_id: Model identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil

            model_dir = self.base_dir / model_id

            if not model_dir.exists():
                logger.warning(f"⚠️ Model not found: {model_id}")
                return False

            # Delete entire model directory
            shutil.rmtree(model_dir)
            logger.info(f"🗑️  Deleted model: {model_id}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete model {model_id}: {e}")
            return False


# Convenience function
def save_trained_model(
    model: Any,
    preprocessor: Any,
    feature_names: list,
    target_column: str,
    task_type: str,
    metrics: Dict[str, float],
    model_name: str,
    **kwargs
) -> Tuple[str, str]:
    """
    Quick save function for trained models.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
        target_column: Target column name
        task_type: "classification" or "regression"
        metrics: Model performance metrics
        model_name: Name of the model (e.g., "XGBoost", "RandomForest")
        **kwargs: Additional metadata fields

    Returns:
        Tuple of (model_id, model_directory_path)
    """
    persistence = ModelPersistence()

    metadata = {
        "model_type": model_name,
        "task_type": task_type,
        "feature_names": feature_names,
        "target_column": target_column,
        "metrics": metrics,
        **kwargs
    }

    return persistence.save_model(model, preprocessor, metadata)


if __name__ == "__main__":
    # Test model persistence
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("MODEL PERSISTENCE TEST")
    print("="*60)

    # Create dummy model and preprocessor for testing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Create and "train" dummy model
    X_dummy = np.random.rand(100, 5)
    y_dummy = np.random.randint(0, 2, 100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)

    preprocessor = StandardScaler()
    preprocessor.fit(X_dummy)

    metadata = {
        "model_type": "RandomForest",
        "task_type": "classification",
        "feature_names": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
        "target_column": "target",
        "metrics": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.79,
            "f1_score": 0.80
        },
        "n_samples": 100,
        "n_features": 5
    }

    # Test save
    print("\n📦 Saving model...")
    persistence = ModelPersistence()
    model_id, model_dir = persistence.save_model(model, preprocessor, metadata)
    print(f"\n✅ Model saved:")
    print(f"   ID: {model_id}")
    print(f"   Directory: {model_dir}")

    # Test load
    print(f"\n📂 Loading model {model_id}...")
    loaded_model, loaded_preprocessor, loaded_metadata = persistence.load_model(model_id)
    print(f"\n✅ Model loaded:")
    print(f"   Type: {loaded_metadata['model_type']}")
    print(f"   Accuracy: {loaded_metadata['metrics']['accuracy']}")
    print(f"   Features: {loaded_metadata['feature_names']}")

    # Test list
    print("\n📋 Listing all models...")
    all_models = persistence.list_models()
    print(f"\n✅ Found {len(all_models)} models:")
    for m in all_models:
        print(f"   - {m['model_id'][:8]}... ({m['model_type']}, {m['metrics']['accuracy']:.2%})")

    # Test delete
    print(f"\n🗑️  Deleting model {model_id}...")
    deleted = persistence.delete_model(model_id)
    print(f"✅ Deleted: {deleted}")

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60 + "\n")
