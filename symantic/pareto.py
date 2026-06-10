#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 23:24:41 2024

@author: muthyala.7
"""
import torch

import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

class pareto:
    
    def __init__(self,rmse,complexity,final_pareto='no',utopia_point=None):

        self.rmse = rmse

        self.complexity = abs(complexity)

        self.final_pareto = final_pareto

        self._utopia_point = utopia_point
        
        

    def pareto_front(self):

        def is_pareto_efficient(costs):
            """O(n log n) Pareto front for 2 objectives via sort-and-scan."""
            n_points = costs.shape[0]
            if n_points == 0:
                return torch.zeros(0, dtype=torch.bool)
            if n_points == 1:
                return torch.ones(1, dtype=torch.bool)

            # Sort by first objective (complexity) ascending
            sorted_idx = torch.argsort(costs[:, 0])
            sorted_costs = costs[sorted_idx]

            is_efficient = torch.zeros(n_points, dtype=torch.bool)
            # Scan left-to-right: a point is Pareto-optimal if its second
            # objective (RMSE) is <= the best seen so far (since it has higher
            # or equal first objective than all predecessors).
            min_obj2 = float('inf')
            for i in range(n_points):
                if sorted_costs[i, 1] <= min_obj2:
                    is_efficient[sorted_idx[i]] = True
                    min_obj2 = sorted_costs[i, 1].item()

            return is_efficient

        # Combine RMSE and complexity into a single tensor
        if self.complexity.numel() == 0:
            return np.array([], dtype=np.intp)
        costs = torch.column_stack((self.complexity, self.rmse))

        # Find the Pareto efficient points
        pareto_efficient_mask = is_pareto_efficient(costs)
        pareto_front = costs[pareto_efficient_mask]

        # Determine the utopia point (best possible values for each objective)
        if self._utopia_point is not None:
            utopia_point = torch.tensor(self._utopia_point)
        else:
            utopia_point = torch.min(costs, axis=0).values

        # Vectorized distance computation
        if pareto_front.shape[0] > 0:
            distances = torch.sqrt(torch.sum((pareto_front - utopia_point.unsqueeze(0)) ** 2, dim=1))
        else:
            distances = torch.empty(0)

        # Sort Pareto-efficient solutions by distance from utopia point
        sorted_indices = torch.argsort(distances)
        sorted_pareto_front = pareto_front[sorted_indices]
        sorted_distances = distances[sorted_indices]

        if self.final_pareto == 'yes':
            plt.figure(figsize=(10, 8))
            plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', label='Pareto front')
            sorted_pareto_front = pareto_front[pareto_front[:, 1].argsort()]
            plt.step(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1], 'r-', where='pre', label='Pareto Line')
            plt.scatter(utopia_point[0], utopia_point[1], c='green', label='Utopia',marker='*',s=100)
            plt.xlabel(r'Complexity = $k \log n$ (bits)',weight='bold')
            plt.ylabel('Accuracy (RMSE)',weight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.title('Pareto Frontier')
            plt.show()

        # Get the original indices of the Pareto efficient solutions
        pareto_indices = np.where(pareto_efficient_mask)[0]

        return pareto_indices