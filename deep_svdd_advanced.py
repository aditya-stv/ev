"""
Deep SVDD (Support Vector Data Description)
===========================================
Research-backed unsupervised anomaly detection.

Based on 2024 research: DSVDD-CAE achieving 99.25% F1-score.
Trains neural network to map normal data into compact hypersphere.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DeepSVDDNet(nn.Module):
    """Deep SVDD Encoder Network."""
    
    def __init__(self, input_dim, latent_dim=16):
        super(DeepSVDDNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)


class DeepSVDD:
    """
    Deep Support Vector Data Description for anomaly detection.
    """
    
    def __init__(self, input_dim, latent_dim=16, nu=0.1):
        """
        Args:
            input_dim: Number of input features
            latent_dim: Dimensionality of latent space
            nu: Anomaly fraction (expected proportion of anomalies)
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nu = nu
        
        self.net = DeepSVDDNet(input_dim, latent_dim)
        self.center = None
        self.radius = None
        self.loss_history = []  # Track training loss for visualization
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        
    def fit(self, X, epochs=50, batch_size=128, lr=0.001, verbose=True):
        """
        Train Deep SVDD on normal data.
        
        Args:
            X: Normal training data (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            verbose: Print training progress
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize center c as mean of initial embeddings
        if verbose:
            print(f"[Deep SVDD] Initializing center...")
        
        self.net.eval()
        with torch.no_grad():
            embeddings = []
            for batch in dataloader:
                batch_x = batch[0]
                emb = self.net(batch_x)
                embeddings.append(emb.cpu().numpy())
            embeddings = np.vstack(embeddings)
            self.center = torch.FloatTensor(np.mean(embeddings, axis=0)).to(self.device)
        
        if verbose:
            print(f"  Center: {self.center.cpu().numpy()[:5]}...")
        
        # Training
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-6)
        
        if verbose:
            print(f"[Deep SVDD] Training for {epochs} epochs...")
        
        self.net.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch_x = batch[0]
                
                optimizer.zero_grad()
                
                # Forward pass
                embeddings = self.net(batch_x)
                
                # Compute distance to center
                dist = torch.sum((embeddings - self.center) ** 2, dim=1)
                
                # Loss: mean squared distance
                loss = torch.mean(dist)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
            
            # Save loss for visualization
            avg_loss = total_loss / len(dataloader)
            self.loss_history.append(avg_loss)
        
        # Set radius as (1-nu) quantile of distances
        self.net.eval()
        with torch.no_grad():
            all_dists = []
            for batch in dataloader:
                batch_x = batch[0]
                emb = self.net(batch_x)
                dist = torch.sum((emb - self.center) ** 2, dim=1)
                all_dists.append(dist.cpu().numpy())
            all_dists = np.concatenate(all_dists)
            self.radius = float(np.quantile(all_dists, 1 - self.nu))
        
        if verbose:
            print(f"  Radius (threshold): {self.radius:.4f}")
            print(f"[Deep SVDD] Training complete!")
        
    def predict(self, X):
        """
        Predict anomalies.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            predictions: 0 for normal, 1 for anomaly
        """
        scores = self.decision_function(X)
        return (scores > self.radius).astype(int)
    
    def decision_function(self, X):
        """
        Compute anomaly scores (distance to center).
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            scores: Anomaly scores (higher = more anomalous)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.net.eval()
        with torch.no_grad():
            embeddings = self.net(X_tensor)
            dist = torch.sum((embeddings - self.center) ** 2, dim=1)
            return dist.cpu().numpy()


if __name__ == "__main__":
    print("Testing Deep SVDD...")
    
    # Create synthetic data
    np.random.seed(42)
    n_normal = 1000
    n_anomaly = 100
    
    # Normal: clustered around origin
    normal_data = np.random.randn(n_normal, 36) * 0.5
    
    # Anomaly: farther from origin
    anomaly_data = np.random.randn(n_anomaly, 36) * 2 + 3
    
    # Train on normal only
    svdd = DeepSVDD(input_dim=36, latent_dim=16, nu=0.1)
    svdd.fit(normal_data, epochs=30, verbose=True)
    
    # Test on both
    normal_preds = svdd.predict(normal_data[:100])
    anomaly_preds = svdd.predict(anomaly_data)
    
    print(f"\nTest Results:")
    print(f"  Normal flagged as anomaly: {normal_preds.sum()}/100 ({normal_preds.mean()*100:.1f}%)")
    print(f"  Anomalies detected: {anomaly_preds.sum()}/100 ({anomaly_preds.mean()*100:.1f}%)")
    
    print("\n✓ Deep SVDD test complete!")
