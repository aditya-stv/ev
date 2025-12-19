"""
Advanced Statistical Feature Engineering for Anomaly Detection
===============================================================
Research-backed feature extraction from limited base metrics.

Based on 2024 findings that 2-3 base features can achieve high
detection rates with proper statistical feature engineering.

References:
- 2024 MDPI study on DDoS detection with limited features
- Statistical analysis: CV, kurtosis, skewness, AR, correlation, KL-div
"""

import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureExtractor:
    """
    Extract 30-40 features from 3 base metrics using advanced statistical methods.
    """
    
    def __init__(self, window_size=20):
        """
        Args:
            window_size: Size of sliding window for temporal features
        """
        self.window_size = window_size
        self.baseline_stats = None
        
    def fit(self, X_normal):
        """
        Compute baseline statistics from normal training data.
        
        Args:
            X_normal: Normal samples (n_samples, 3) - [cycles, instructions, branches]
        """
        print("[Feature Extractor] Computing baseline statistics...")
        
        self.baseline_stats = {
            'mean': np.mean(X_normal, axis=0),
            'std': np.std(X_normal, axis=0),
            'distribution': X_normal  # Store for KL-divergence
        }
        
        print(f"  Baseline: mean={self.baseline_stats['mean']}, std={self.baseline_stats['std']}")
        
    def extract_features(self, data_sequence):
        """
        Extract all features from a sequence of measurements.
        
        Args:
            data_sequence: Array of shape (seq_length, 3) - time series of measurements
            
        Returns:
            feature_vector: Array of ~35 features
        """
        features = {}
        
        # Ensure we have enough data
        if len(data_sequence) < self.window_size:
            # Pad with last value
            pad_length = self.window_size - len(data_sequence)
            data_sequence = np.vstack([
                data_sequence,
                np.repeat(data_sequence[-1:], pad_length, axis=0)
            ])
        
        # Use last window_size points
        window = data_sequence[-self.window_size:]
        
        # ===== BASIC STATISTICS =====
        features['mean_cycles'] = np.mean(window[:, 0])
        features['mean_instructions'] = np.mean(window[:, 1])
        features['mean_branches'] = np.mean(window[:, 2])
        
        features['std_cycles'] = np.std(window[:, 0])
        features['std_instructions'] = np.std(window[:, 1])
        features['std_branches'] = np.std(window[:, 2])
        
        # ===== ROBUST STATISTICS (IQR) =====
        # IQR is less sensitive to outliers than STD, helping distinguish legitimate spikes
        features['iqr_cycles'] = stats.iqr(window[:, 0])
        features['iqr_instructions'] = stats.iqr(window[:, 1])
        features['iqr_branches'] = stats.iqr(window[:, 2])
        
        # ===== ENTROPY (Chaos Metric) =====
        # Attacks often introduce chaos (high entropy) or repetitive patterns (low entropy)
        for i, name in enumerate(['cycles', 'instructions', 'branches']):
            # Histogram-based entropy
            hist, _ = np.histogram(window[:, i], bins=5, density=True)
            features[f'entropy_{name}'] = stats.entropy(hist + 1e-10)

        # ===== COEFFICIENT OF VARIATION (Research-backed) =====
        features['cv_cycles'] = features['std_cycles'] / (features['mean_cycles'] + 1e-10)
        features['cv_instructions'] = features['std_instructions'] / (features['mean_instructions'] + 1e-10)
        features['cv_branches'] = features['std_branches'] / (features['mean_branches'] + 1e-10)
        
        # ===== DISTRIBUTION SHAPE (Kurtosis & Skewness) =====
        features['kurt_cycles'] = stats.kurtosis(window[:, 0])
        features['kurt_instructions'] = stats.kurtosis(window[:, 1])
        features['kurt_branches'] = stats.kurtosis(window[:, 2])
        
        features['skew_cycles'] = stats.skew(window[:, 0])
        features['skew_instructions'] = stats.skew(window[:, 1])
        features['skew_branches'] = stats.skew(window[:, 2])
        
        # ===== AUTOREGRESSION COEFFICIENTS =====
        for i, name in enumerate(['cycles', 'instructions', 'branches']):
            ar_coef = self._compute_ar1(window[:, i])
            features[f'ar1_{name}'] = ar_coef
        
        # ===== CROSS-CORRELATION (Research-backed) =====
        features['corr_cycles_instructions'] = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
        features['corr_cycles_branches'] = np.corrcoef(window[:, 0], window[:, 2])[0, 1]
        features['corr_instructions_branches'] = np.corrcoef(window[:, 1], window[:, 2])[0, 1]
        
        # ===== RATE OF CHANGE (Derivatives) =====
        for i, name in enumerate(['cycles', 'instructions', 'branches']):
            diff = np.diff(window[:, i])
            features[f'roc_mean_{name}'] = np.mean(diff)
            features[f'roc_std_{name}'] = np.std(diff)
        
        # ===== KL-DIVERGENCE from baseline (Research-backed) =====
        if self.baseline_stats is not None:
            for i, name in enumerate(['cycles', 'instructions', 'branches']):
                kl_div = self._compute_kl_divergence(window[:, i], i)
                features[f'kl_div_{name}'] = kl_div
        
        # ===== FREQUENCY DOMAIN (FFT components) =====
        for i, name in enumerate(['cycles', 'instructions', 'branches']):
            freq_features = self._extract_frequency_features(window[:, i])
            features[f'fft_dominant_{name}'] = freq_features['dominant_freq']
            features[f'fft_energy_{name}'] = freq_features['energy']
            features[f'fft_low_band_{name}'] = freq_features['low_band_energy']
            features[f'fft_high_band_{name}'] = freq_features['high_band_energy']
        
        # Convert to array
        feature_array = np.array(list(features.values()))
        
        # Handle any NaN/Inf
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_array
    
    def _compute_ar1(self, series):
        """Compute AR(1) coefficient."""
        if len(series) < 2:
            return 0.0
        
        # Simple AR(1): correlation between x_t and x_{t-1}
        return np.corrcoef(series[:-1], series[1:])[0, 1] if len(series) > 1 else 0.0
    
    def _compute_kl_divergence(self, sample, metric_idx):
        """Compute KL-divergence from baseline distribution."""
        if self.baseline_stats is None:
            return 0.0
        
        try:
            # Create histograms
            baseline_data = self.baseline_stats['distribution'][:, metric_idx]
            
            # Use same bins for both
            bins = np.linspace(
                min(baseline_data.min(), sample.min()),
                max(baseline_data.max(), sample.max()),
                20
            )
            
            p, _ = np.histogram(baseline_data, bins=bins, density=True)
            q, _ = np.histogram(sample, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            p = p + 1e-10
            q = q + 1e-10
            
            # Normalize
            p = p / p.sum()
            q = q / q.sum()
            
            # KL divergence
            kl = entropy(p, q)
            
            return kl if np.isfinite(kl) else 0.0
        except:
            return 0.0
    
    def _extract_frequency_features(self, series):
        """Extract frequency domain features using FFT."""
        # Compute FFT
        fft_vals = fft(series)
        fft_magnitude = np.abs(fft_vals)
        
        # Get dominant frequency (exclude DC component)
        dominant_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
        dominant_freq = dominant_idx / len(series)
        
        # Total energy (sum of squared magnitudes)
        energy = np.sum(fft_magnitude ** 2)
        
        # Band energies (Low: <25%, High: >25% of spectrum)
        n = len(series)
        low_band_energy = np.sum(fft_magnitude[1:n//4] ** 2)
        high_band_energy = np.sum(fft_magnitude[n//4:n//2] ** 2)
        
        return {
            'dominant_freq': dominant_freq,
            'energy': energy,
            'low_band_energy': low_band_energy,
            'high_band_energy': high_band_energy
        }


def test_feature_extractor():
    """Test the feature extractor."""
    print("Testing Advanced Feature Extractor...")
    
    # Create synthetic normal data
    np.random.seed(42)
    n_samples = 100
    normal_data = np.random.randn(n_samples, 3) * [1000, 500, 200] + [5000, 2000, 800]
    
    # Initialize and fit
    extractor = AdvancedFeatureExtractor(window_size=20)
    extractor.fit(normal_data)
    
    # Extract features from a sequence
    test_sequence = normal_data[-25:]  # Last 25 points
    features = extractor.extract_features(test_sequence)
    
    print(f"\nExtracted {len(features)} features:")
    print(f"  Shape: {features.shape}")
    print(f"  Range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Mean: {features.mean():.3f}")
    print(f"  Std: {features.std():.3f}")
    
    # Test with anomalous data
    anomaly_data = normal_data[-25:] * 2.5  # Amplify
    anomaly_features = extractor.extract_features(anomaly_data)
    
    # Compare
    diff = np.abs(features - anomaly_features).mean()
    print(f"\nNormal vs Anomaly feature difference: {diff:.3f}")
    
    print("\n✓ Feature extractor test complete!")


if __name__ == "__main__":
    test_feature_extractor()
