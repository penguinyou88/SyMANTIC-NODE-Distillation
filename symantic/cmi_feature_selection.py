#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:28:24 2024

@author: muthyala.7
"""

import torch
import torch.nn.functional as F
from torch.special import digamma
from typing import Optional, Tuple, List

class InfoTheoryEstimators:
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the estimators with a specified device.
        
        Args:
            device: torch.device to use for computations
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _add_noise(self, x: torch.Tensor, intensity: float = 1e-10) -> torch.Tensor:
        """Add small noise to break degeneracy"""
        return x + intensity * torch.rand_like(x, device=self.device)
    
    def _build_distance_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise maximum norm (Chebyshev) distances between points.
        Uses batch operations for better performance.
        """
        n_samples = x.size(0)
        # Reshape x for broadcasting
        x_expanded_1 = x.unsqueeze(1)  # Shape: (n, 1, features)
        x_expanded_2 = x.unsqueeze(0)  # Shape: (1, n, features)
        
        # Compute distances using broadcasting
        distances = torch.max(torch.abs(x_expanded_1 - x_expanded_2), dim=2)[0]
        return distances
    
    def _get_knn_distances(self, distances: torch.Tensor, k: int) -> torch.Tensor:
        """Get the k-th nearest neighbor distances"""
        # Sort distances and get the k-th nearest neighbor (k+1 because first one is self)
        knn_distances = torch.topk(distances, k=k+1, dim=1, largest=False)[0][:, k]
        return knn_distances
    
    def _count_neighbors_within_radius(self, distances: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """Count number of neighbors within radius for each point"""
        return (distances <= radius.unsqueeze(1)).sum(dim=1).float()
    
    def _avgdigamma(self, points: torch.Tensor, dvec: torch.Tensor) -> torch.Tensor:
        """Compute average digamma value"""
        distances = self._build_distance_matrix(points)
        dvec = dvec.unsqueeze(1) - 1e-15
        counts = (distances <= dvec).sum(dim=1).float()
        return digamma(counts).mean()
    
    def entropy(self, x: torch.Tensor, k: int = 3, base: float = 2.0) -> torch.Tensor:
        """
        Estimate the entropy of x using k-nearest neighbors approach
        
        Args:
            x: Input tensor of shape (n_samples, n_features)
            k: Number of nearest neighbors
            base: Logarithm base for entropy calculation
        
        Returns:
            Estimated entropy value
        """
        x = self._add_noise(x)
        n_elements, n_features = x.shape
        
        # Get k-nn distances
        distances = self._build_distance_matrix(x)
        nn_distances = self._get_knn_distances(distances, k)
        
        # Compute entropy estimate
        const = digamma(torch.tensor(n_elements)) - digamma(torch.tensor(k)) + n_features * torch.log(torch.tensor(2.0))
        entropy_est = (const + n_features * torch.log(nn_distances).mean()) / torch.log(torch.tensor(base))
        
        return entropy_est
    
    def mutual_information(self, x: torch.Tensor, y: torch.Tensor, k: int = 3, base: float = 2.0) -> torch.Tensor:
        """
        Estimate the mutual information between x and y
        
        Args:
            x: First input tensor of shape (n_samples, n_features_x)
            y: Second input tensor of shape (n_samples, n_features_y)
            k: Number of nearest neighbors
            base: Logarithm base for MI calculation
        
        Returns:
            Estimated mutual information value
        """
        x = self._add_noise(x)
        y = self._add_noise(y)
        
        # Concatenate points
        points = torch.cat([x, y], dim=1)
        
        # Get k-nn distances in joint space
        distances = self._build_distance_matrix(points)
        dvec = self._get_knn_distances(distances, k)
        
        # Compute mutual information estimate
        a = self._avgdigamma(x, dvec)
        b = self._avgdigamma(y, dvec)
        c = digamma(torch.tensor(k, device=self.device))
        d = digamma(torch.tensor(x.size(0), device=self.device))
        
        mi = (-a - b + c + d) / torch.log(torch.tensor(base, device=self.device))
        return mi
    
    def conditional_mutual_information(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, 
                                     k: int = 3, base: float = 2.0) -> torch.Tensor:
        """
        Estimate the conditional mutual information between x and y given z
        
        Args:
            x: First input tensor of shape (n_samples, n_features_x)
            y: Second input tensor of shape (n_samples, n_features_y)
            z: Conditioning tensor of shape (n_samples, n_features_z)
            k: Number of nearest neighbors
            base: Logarithm base for CMI calculation
        
        Returns:
            Estimated conditional mutual information value
        """
        x = self._add_noise(x)
        y = self._add_noise(y)
        z = self._add_noise(z)
        
        points = torch.cat([x, y, z], dim=1)
        distances = self._build_distance_matrix(points)
        dvec = self._get_knn_distances(distances, k)
        
        xz = torch.cat([x, z], dim=1)
        yz = torch.cat([y, z], dim=1)
        
        a = self._avgdigamma(xz, dvec)
        b = self._avgdigamma(yz, dvec)
        c = self._avgdigamma(z, dvec)
        d = digamma(torch.tensor(k, device=self.device))
        
        cmi = (-a - b + c + d) / torch.log(torch.tensor(base, device=self.device))
        return cmi
    
    def select_features(self, X: torch.Tensor, Y: torch.Tensor, 
                       num_features: Optional[int] = None, 
                       threshold: Optional[float] = None,
                       k: int = 3, base: float = 2.0) -> List[int]:
        """
        Select features based on mutual information and conditional mutual information
        
        Args:
            X: Input features tensor of shape (n_samples, n_features)
            Y: Target tensor of shape (n_samples, 1)
            num_features: Maximum number of features to select
            threshold: Minimum information threshold for feature selection
            k: Number of nearest neighbors
            base: Logarithm base for information calculations
        
        Returns:
            List of selected feature indices
        """
        n_samples, n_features = X.shape
        selected_vars = []
        remaining_vars = list(range(n_features))
        
        # Initial MI scores computation (in parallel)
        mi_scores = torch.zeros(n_features, device=self.device)
        for i in remaining_vars:
            mi_scores[i] = self.mutual_information(
                X[:, [i]].reshape(-1, 1),
                Y,
                k=k,
                base=base
            )
        
        while remaining_vars:
            # Select best variable
            best_var = remaining_vars[torch.argmax(mi_scores[remaining_vars]).item()]
            selected_vars.append(best_var)
            remaining_vars.remove(best_var)
            
            # Check stopping conditions
            if num_features and len(selected_vars) >= num_features:
                break
            if not remaining_vars:
                break
                
            # Update scores with CMI
            selected_tensor = X[:, selected_vars]
            conditional_scores = torch.zeros(len(remaining_vars), device=self.device)
            
            for idx, var in enumerate(remaining_vars):
                x_current = torch.cat([selected_tensor, X[:, [var]]], dim=1)
                cmi_score = self.conditional_mutual_information(
                    x_current,
                    Y,
                    selected_tensor,
                    k=k,
                    base=base
                )
                conditional_scores[idx] = cmi_score
            
            max_cmi = conditional_scores.max()
            if threshold and max_cmi < threshold:
                break
                
            mi_scores = torch.zeros_like(mi_scores)
            for idx, var in enumerate(remaining_vars):
                mi_scores[var] = conditional_scores[idx]
        
        return selected_vars
# =============================================================================

# Example usage:
# import warnings
# warnings.filterwarnings('ignore')
# if __name__ == "__main__":
#     for i in range(2):
#       n_samples, n_features = 100, 1000
#       X = torch.rand(n_samples, n_features)
#       Y = ((X[:, 1] + torch.log(X[:, 3])) / (X[:, 5] + X[:, 6])).reshape(-1, 1)
      
#       # Initialize estimator
#       estimator = InfoTheoryEstimators()
      
#       # Select features
#       print(torch.tensor(X).shape, torch.tensor(Y).reshape(-1, 1).shape)
#       selected_features = estimator.select_features(torch.tensor(X), torch.tensor(Y).reshape(-1, 1), num_features=100)
#       print("Selected feature indices:", selected_features)

#       from minepy import MINE,cstats

#       mine1=[]
#       for i in range(X.shape[1]):
#         mine=MINE(alpha=0.6, c=15, est="mic_approx")
#         mine.compute_score(X[:,i].numpy(),Y.flatten().numpy())
#         mine1.append(mine.mic())
#       k,z = torch.topk(torch.tensor(mine1),k=100)
#       print('Minepy indices:',z.numpy().tolist())

#       print('\n \n  \n')  


      
# # =============================================================================

