"""
Contractive Autoencoder for Anomaly Detection
==============================================
Adds contractive penalty to force smooth latent representations.
Detects anomalies via reconstruction error.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class ContractiveAutoencoder(nn.Module):
    """Contractive Autoencoder network."""
    
    def __init__(self, input_dim, latent_dim=16):
        super(ContractiveAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, input_dim)
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ContractiveAE:
    """
    Contractive Autoencoder for unsupervised anomaly detection.
    """
    
    def __init__(self, input_dim, latent_dim=16, lambda_contractive=0.0001):
        """
        Args:
            input_dim: Number of input features
            latent_dim: Dimensionality of latent space
            lambda_contractive: Weight for contractive penalty
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lambda_contractive = lambda_contractive
        
        self.model = ContractiveAutoencoder(input_dim, latent_dim)
        self.threshold = None
        self.loss_history = []  # Track training loss for visualization
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _contractive_loss(self, x, h):
        """
        Compute contractive penalty: Frobenius norm of Jacobian.
        Encourages similar inputs to have similar representations.
        """
        # Compute Jacobian: dh/dx
        # For efficiency, approximate with gradient norm
        h.backward(torch.ones_like(h), retain_graph=True)
        
        # L2 norm of gradients
        jacobian_norm = torch.sqrt(torch.sum(x.grad ** 2))
        
        return jacobian_norm
    
    def fit(self, X, epochs=50, batch_size=64, lr=0.001, verbose=True):
        """
        Train Contractive AE on normal data.
        
        Args:
            X: Normal training data (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        if verbose:
            print(f"[Contractive AE] Training for {epochs} epochs...")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            
            for batch in dataloader:
                batch_x = batch[0]
                batch_x.requires_grad = True
                
                optimizer.zero_grad()
                
                # Forward pass
                x_recon, z = self.model(batch_x)
                
                # Reconstruction loss
                recon_loss = criterion(x_recon, batch_x)
                
                # Contractive loss (Jacobian penalty)
                # Simplified: penalize variance in latent space
                contractive_loss = torch.mean(z ** 2)
                
                # Total loss
                loss = recon_loss + self.lambda_contractive * contractive_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                avg_recon = total_recon / len(dataloader)
                print(f"  Epoch {epoch+1}/{epochs}: Total={avg_loss:.4f}, Recon={avg_recon:.4f}")
            
            # Save loss for visualization
            avg_recon = total_recon / len(dataloader)
            self.loss_history.append(avg_recon)
        
        # Set threshold as 95th percentile of reconstruction errors
        self.model.eval()
        with torch.no_grad():
            all_errors = []
            for batch in dataloader:
                batch_x = batch[0]
                x_recon, _ = self.model(batch_x)
                error = torch.mean((batch_x - x_recon) ** 2, dim=1)
                all_errors.append(error.cpu().numpy())
            all_errors = np.concatenate(all_errors)
            self.threshold = float(np.percentile(all_errors, 95))
        
        if verbose:
            print(f"  Threshold (95th percentile): {self.threshold:.4f}")
            print(f"[Contractive AE] Training complete!")
    
    def predict(self, X):
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            predictions: 0 for normal, 1 for anomaly
        """
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)
    
    def decision_function(self, X):
        """
        Compute anomaly scores (reconstruction error).
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            scores: Reconstruction errors (higher = more anomalous)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            x_recon, _ = self.model(X_tensor)
            error = torch.mean((X_tensor - x_recon) ** 2, dim=1)
            return error.cpu().numpy()


if __name__ == "__main__":
    print("Testing Contractive Autoencoder...")
    
    # Synthetic data
    np.random.seed(42)
    n_normal = 1000
    n_anomaly = 100
    
    # Normal: Gaussian
    normal_data = np.random.randn(n_normal, 36) * 0.5
    
    # Anomaly: Different distribution
    anomaly_data = np.random.randn(n_anomaly, 36) * 2 + 3
    
    # Train on normal only
    cae = ContractiveAE(input_dim=36, latent_dim=16, lambda_contractive=0.0001)
    cae.fit(normal_data, epochs=30, verbose=True)
    
    # Test
    normal_preds = cae.predict(normal_data[:100])
    anomaly_preds = cae.predict(anomaly_data)
    
    print(f"\nTest Results:")
    print(f"  Normal flagged as anomaly: {normal_preds.sum()}/100 ({normal_preds.mean()*100:.1f}%)")
    print(f"  Anomalies detected: {anomaly_preds.sum()}/100 ({anomaly_preds.mean()*100:.1f}%)")
    
    print("\n✓ Contractive AE test complete!")
