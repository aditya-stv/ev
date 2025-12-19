

import torch
import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, Tuple
from pathlib import Path

from advanced_features import AdvancedFeatureExtractor
from deep_svdd_advanced import DeepSVDD
from contractive_ae import ContractiveAE
from sklearn.preprocessing import StandardScaler


class AMTEADProduction:
    """Production-ready AMTEAD ensemble for anomaly detection."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize AMTEAD production system.
        
        Args:
            config: Configuration dictionary with hyperparameters
        """
        self.config = config or self._default_config()
        
        # Components
        self.feature_extractor = None
        self.scaler = None
        self.deep_svdd = None
        self.cae = None
        
        # Ensemble parameters (optimized)
        self.w_svdd = 0.30
        self.w_cae = 0.70
        self.threshold = None
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'feature_extractor': {
                'window_size': 20
            },
            'deep_svdd': {
                'latent_dim': 16,
                'nu': 0.15,
                'epochs': 60,
                'batch_size': 64,
                'lr': 0.001
            },
            'cae': {
                'latent_dim': 16,
                'lambda_contractive': 0.0001,
                'epochs': 60,
                'batch_size': 64,
                'lr': 0.001
            },
            'ensemble': {
                'w_svdd': 0.30,
                'w_cae': 0.70,
                'threshold_percentile': 60
            }
        }
    
    def fit(self, sequences_normal: list, verbose: bool = True):
        """
        Train the system on normal traffic sequences.
        
        Args:
            sequences_normal: List of normal traffic sequences (each shape: [25, 3])
            verbose: Print training progress
        """
        if verbose:
            print("[AMTEAD] Training on normal traffic...")
            print(f"  Samples: {len(sequences_normal)}")
        
        # 1. Feature extraction
        if verbose:
            print("\n[1/4] Fitting feature extractor...")
        
        self.feature_extractor = AdvancedFeatureExtractor(
            window_size=self.config['feature_extractor']['window_size']
        )
        
        # Prepare baseline data
        normal_base = np.vstack([seq[-20:] for seq in sequences_normal])
        self.feature_extractor.fit(normal_base)
        
        # Extract features
        X_features = []
        for seq in sequences_normal:
            features = self.feature_extractor.extract_features(seq)
            X_features.append(features)
        
        X = np.array(X_features)
        
        if verbose:
            print(f"  Extracted {X.shape[1]} features")
        
        # 2. Scaling
        if verbose:
            print("\n[2/4] Scaling features...")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 3. Train Deep SVDD
        if verbose:
            print("\n[3/4] Training Deep SVDD...")
        
        svdd_config = self.config['deep_svdd']
        self.deep_svdd = DeepSVDD(
            input_dim=X.shape[1],
            latent_dim=svdd_config['latent_dim'],
            nu=svdd_config['nu']
        )
        self.deep_svdd.fit(
            X_scaled,
            epochs=svdd_config['epochs'],
            batch_size=svdd_config['batch_size'],
            lr=svdd_config['lr'],
            verbose=verbose
        )
        
        # 4. Train Contractive AE
        if verbose:
            print("\n[4/4] Training Contractive AE...")
        
        cae_config = self.config['cae']
        self.cae = ContractiveAE(
            input_dim=X.shape[1],
            latent_dim=cae_config['latent_dim'],
            lambda_contractive=cae_config['lambda_contractive']
        )
        self.cae.fit(
            X_scaled,
            epochs=cae_config['epochs'],
            batch_size=cae_config['batch_size'],
            lr=cae_config['lr'],
            verbose=verbose
        )
        
        # 5. Set ensemble threshold
        ensemble_scores = self._compute_ensemble_scores(X_scaled)
        percentile = self.config['ensemble']['threshold_percentile']
        self.threshold = np.percentile(ensemble_scores, percentile)
        
        if verbose:
            print(f"\n[AMTEAD] Training complete!")
            print(f"  Ensemble threshold: {self.threshold:.4f} (P{percentile})")
    
    def _compute_ensemble_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble anomaly scores."""
        # Get scores from both models
        svdd_scores = self.deep_svdd.decision_function(X)
        cae_scores = self.cae.decision_function(X)
        
        # Normalize to [0, 1]
        svdd_norm = (svdd_scores - svdd_scores.min()) / (svdd_scores.max() - svdd_scores.min() + 1e-10)
        cae_norm = (cae_scores - cae_scores.min()) / (cae_scores.max() - cae_scores.min() + 1e-10)
        
        # Weighted ensemble
        ensemble = self.w_svdd * svdd_norm + self.w_cae * cae_norm
        
        return ensemble
    
    def predict(self, sequence: np.ndarray) -> Tuple[int, float, Dict]:
        """
        Predict if a sequence is an attack.
        
        Args:
            sequence: Traffic sequence (shape: [25, 3])
            
        Returns:
            prediction: 0 for normal, 1 for attack
            confidence: Confidence score (0-1)
            details: Detailed scores from each component
        """
        # Extract features
        features = self.feature_extractor.extract_features(sequence)
        X = features.reshape(1, -1)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Get ensemble score
        ensemble_score = self._compute_ensemble_scores(X_scaled)[0]
        
        # Prediction
        prediction = 1 if ensemble_score > self.threshold else 0
        
        # Confidence (distance from threshold)
        confidence = abs(ensemble_score - self.threshold) / self.threshold
        confidence = min(confidence, 1.0)
        
        # Details
        svdd_score = self.deep_svdd.decision_function(X_scaled)[0]
        cae_score = self.cae.decision_function(X_scaled)[0]
        
        details = {
            'ensemble_score': float(ensemble_score),
            'threshold': float(self.threshold),
            'svdd_score': float(svdd_score),
            'cae_score': float(cae_score),
            'svdd_weight': self.w_svdd,
            'cae_weight': self.w_cae
        }
        
        return prediction, confidence, details
    
    def predict_with_mitigation(self, sequence: np.ndarray) -> Tuple[int, float, str, Dict]:
        """
        Predict with mitigation action (PRODUCTION-READY).
        
        Args:
            sequence: Traffic sequence (shape: [25, 3])
            
        Returns:
            prediction: 0 for normal, 1 for attack
            confidence: Confidence score (0-1)
            mitigation_action: One of ['ALLOW', 'MONITOR', 'THROTTLE_25', 'THROTTLE_75', 'BLOCK']
            details: Detailed scores and reasoning
        """
        prediction, confidence, details = self.predict(sequence)
        
        # Determine mitigation action
        if prediction == 0:
            # Normal traffic - allow
            mitigation_action = 'ALLOW'
            details['mitigation_reason'] = 'Traffic classified as normal'
        else:
            # Attack detected - apply graduated response based on confidence
            ensemble_score = details['ensemble_score']
            threshold = details['threshold']
            
            # Calculate relative confidence above threshold
            distance_from_threshold = ensemble_score - threshold
            threshold_range = 1.0 - threshold
            relative_confidence = distance_from_threshold / threshold_range if threshold_range > 0 else 1.0
            
            # AGGRESSIVE PRODUCTION LOGIC
            if relative_confidence > 0.70:
                mitigation_action = 'BLOCK'
                details['mitigation_reason'] = 'Very high confidence attack - hard block'
            elif relative_confidence > 0.45:
                mitigation_action = 'THROTTLE_75'
                details['mitigation_reason'] = 'High confidence attack - throttle 75%'
            elif relative_confidence > 0.20:
                mitigation_action = 'THROTTLE_25'
                details['mitigation_reason'] = 'Medium confidence attack - throttle 25%'
            else:
                mitigation_action = 'MONITOR'
                details['mitigation_reason'] = 'Low confidence attack - monitor only'
            
            details['relative_confidence'] = float(relative_confidence)
        
        return prediction, confidence, mitigation_action, details
    
    def predict_batch(self, sequences: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict for multiple sequences.
        
        Args:
            sequences: List of traffic sequences
            
        Returns:
            predictions: Array of predictions (0/1)
            confidences: Array of confidence scores
        """
        predictions = []
        confidences = []
        
        for seq in sequences:
            pred, conf, _ = self.predict(seq)
            predictions.append(pred)
            confidences.append(conf)
        
        return np.array(predictions), np.array(confidences)
    
    def save(self, filepath: str):
        """Save the trained model."""
        save_dict = {
            'config': self.config,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor,
            'deep_svdd': self.deep_svdd,
            'cae': self.cae,
            'w_svdd': self.w_svdd,
            'w_cae': self.w_cae,
            'threshold': self.threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"[AMTEAD] Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Create instance
        model = cls(config=save_dict['config'])
        model.scaler = save_dict['scaler']
        model.feature_extractor = save_dict['feature_extractor']
        model.deep_svdd = save_dict['deep_svdd']
        model.cae = save_dict['cae']
        model.w_svdd = save_dict['w_svdd']
        model.w_cae = save_dict['w_cae']
        model.threshold = save_dict['threshold']
        
        print(f"[AMTEAD] Model loaded from {filepath}")
        return model


# Example usage
if __name__ == "__main__":
    print("AMTEAD Production - Example Usage\n")
    
    # Create synthetic normal sequences for demo
    np.random.seed(42)
    n_sequences = 100
    normal_sequences = [
        np.random.randn(25, 3) * [100, 50, 20] + [500, 200, 80]
        for _ in range(n_sequences)
    ]
    
    # Train
    print("Training...")
    amtead = AMTEADProduction()
    amtead.fit(normal_sequences, verbose=True)
    
    # Test prediction
    print("\n" + "="*50)
    print("Testing Predictions\n")
    
    # Normal sample
    normal_test = normal_sequences[0]
    pred, conf, details = amtead.predict(normal_test)
    print(f"Normal Sample:")
    print(f"  Prediction: {'Attack' if pred == 1 else 'Normal'}")
    print(f"  Confidence: {conf:.2f}")
    print(f"  Score: {details['ensemble_score']:.4f} (threshold: {details['threshold']:.4f})")
    
    # Anomaly sample
    anomaly_test = normal_sequences[0] * 3  # Amplified
    pred, conf, details = amtead.predict(anomaly_test)
    print(f"\nAnomaly Sample:")
    print(f"  Prediction: {'Attack' if pred == 1 else 'Normal'}")
    print(f"  Confidence: {conf:.2f}")
    print(f"  Score: {details['ensemble_score']:.4f} (threshold: {details['threshold']:.4f})")
    
    # Save model
    print("\n" + "="*50)
    amtead.save('amtead_model.pkl')
    
    # Load and test
    loaded_model = AMTEADProduction.load('amtead_model.pkl')
    pred, conf, _ = loaded_model.predict(normal_test)
    print(f"\nLoaded model prediction: {'Attack' if pred == 1 else 'Normal'} (conf: {conf:.2f})")
    
    print("\n✓ Production system ready!")
