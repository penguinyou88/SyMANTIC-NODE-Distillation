# %%
import torch
import warnings
import numpy as np 
warnings.filterwarnings('ignore')

class MINE(torch.nn.Module):
    def __init__(self, alpha=0.6, c=15):
        """
        Initialize MINE parameters.
        :param alpha: Controls grid partitions. B = max(n^alpha, 4)
        :param c: Controls clump factor in partitioning.
        """
        super(MINE, self).__init__()
        self.alpha = alpha
        self.c = c

    def _grid_partition(self, x, y, bins):
        """
        Create a 2D grid partition for the data, returning a histogram using vectorized operations.
        :param x: Input tensor x.
        :param y: Input tensor y.
        :param bins: Number of bins for partitioning.
        :return: 2D histogram for x and y.
        """
        x_edges = torch.linspace(torch.min(x), torch.max(x), bins+1)
        y_edges = torch.linspace(torch.min(y), torch.max(y), bins+1)
        
        # Digitize data points to create the 2D histogram using tensor operations
        x_bin_idx = torch.bucketize(x, x_edges, right=True) - 1  # Bin index for x
        y_bin_idx = torch.bucketize(y, y_edges, right=True) - 1  # Bin index for y

        # Ensure bin indices stay within bounds
        x_bin_idx = torch.clamp(x_bin_idx, 0, bins-1)
        y_bin_idx = torch.clamp(y_bin_idx, 0, bins-1)

        # Use scatter_add to accumulate counts for the 2D histogram
        hist = torch.zeros((bins, bins), dtype=torch.float32, device=x.device)
        
        # Compute the 2D histogram using scatter_add based on both indices simultaneously
        indices = x_bin_idx * bins + y_bin_idx  # Flattened 2D indices
        flattened_hist = hist.view(-1)  # Flatten the histogram for scatter_add
        flattened_hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))

        return hist

    def _mutual_information(self, hist):
        """
        Calculate mutual information for a given joint histogram using tensor operations.
        :param hist: Joint histogram.
        :return: Mutual Information (MI) value.
        """
        pxy = hist / torch.sum(hist)
        px = torch.sum(pxy, dim=1)  # marginal for x
        py = torch.sum(pxy, dim=0)  # marginal for y

        px_py = torch.outer(px, py)  # Independent distribution
        non_zero = pxy > 0  # Only calculate on non-zero entries

        return torch.sum(pxy[non_zero] * torch.log(pxy[non_zero] / px_py[non_zero]))

    def _max_information(self, x, y):
        """
        Compute maximal mutual information across different grid partitions.
        :param x: Input tensor x.
        :param y: Input tensor y.
        :return: Maximal Information Coefficient (MIC) without bounding.
        """
        n = x.shape[0]
        max_bins = int(min(n, max(4, n ** self.alpha)))
        mic = 0

        for bins in range(2, max_bins):
            hist = self._grid_partition(x, y, bins)
            mi = self._mutual_information(hist)
            mic = max(mic, mi.item())
        return mic

    def compute_score(self, x, y):
        """
        Compute the MIC score between two variables x and y without bounding it between 0 and 1.
        :param x: Input tensor x.
        :param y: Input tensor y.
        """
        return self._max_information(x, y)

    def tic(self, x, y):
        """
        Returns the Total Information Coefficient (TIC).
        :param x: Input tensor x.
        :param y: Input tensor y.
        """
        n = x.shape[0]
        hist = self._grid_partition(x, y, int(min(n, max(4, n ** self.alpha))))
        tic = self._mutual_information(hist)
        return tic

# import numpy as np

# x = np.random.uniform(1,5,(100,1000))
# y = (x[:,0]+x[:,3]/(x[:,4]-x[:,7]))

# #10*(x[:,0]/(x[:,1]*(x[:,2]+x[:,3]))) #+ np.random.normal(0,0.05,2000)


               
# print('########################################')
# x1 = torch.tensor(x)
# y1 = torch.tensor(y)

# sorted_indices = torch.argsort(x1[:, 0])  # Sort based on the first column
# x1_sorted = x1[sorted_indices]  # Apply sorting to x
# y1_sorted = y1[sorted_indices]

# mine = MINE(alpha=0.6, c=15)
# mic1= torch.empty(0,)
# for i in range(x.shape[1]):
#     mic_score = mine.compute_score(x1_sorted[:,i], y1_sorted)

#     mic1 = torch.cat((mic1,torch.tensor(mic_score).reshape(1,-1)),dim=1)


# from minepy import MINE,cstats

# mine1=[]
# for i in range(x.shape[1]):
#     mine=MINE(alpha=0.6, c=15, est="mic_approx")
#     mine.compute_score(x[:,i],y)
#     mine1.append(mine.mic())

# print(torch.topk(torch.tensor(mine1),k=100))


# print(torch.topk(mic1,k=100))

# print()

# import matplotlib.pyplot as plt

# import seaborn as sns
# import numpy as np
# import torch

# # Assuming the necessary data from your provided code is available:
# # `mic1` (your method) and `mine1` (minepy's method).

# # Create a scatter plot comparing the scores of both methods
# plt.figure(figsize=(8, 6))
# plt.scatter(mine1, mic1, alpha=0.7)
# plt.title("Comparison of MIC Scores: Your Method vs MinePy")
# plt.xlabel("MinePy MIC Scores")
# plt.ylabel("Your MIC Scores")
# plt.grid(True)
# plt.show()

# # Create histograms to compare the distributions of the scores
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.hist(mine1, bins=20, color='b', alpha=0.7, label='MinePy MIC')
# plt.title("MinePy MIC Score Distribution")
# plt.xlabel("MIC Score")
# plt.ylabel("Frequency")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.hist(mic1, bins=20, color='g', alpha=0.7, label='MIC_torch')
# plt.title("MIC torch Distribution")
# plt.xlabel("MIC Score")
# plt.ylabel("Frequency")
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Box plots to compare the distributionsS
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=[mine1, mic1], palette="Set2")
# plt.xticks([0, 1], ['MinePy MIC', 'MIC torch'])
# plt.title("Comparison of MIC Scores Using Box Plot")
# plt.ylabel("MIC Score")
# plt.show()



# # Difference Plot
# differences = np.array(mine1) - np.array(mic1)

# plt.figure(figsize=(8, 6))
# plt.scatter(np.arange(0,x.shape[1]),differences, color='r', alpha=0.7)
# plt.title("Difference in MIC Scores: MinePy vs Your Method")
# plt.xlabel("Feature Index")
# plt.ylabel("Score Difference (MinePy - Your Method)")
# plt.grid(True)
# plt.show()

# # %%
