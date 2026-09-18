
# %%

import torch
from scipy.spatial.distance import pdist, squareform

import warnings
warnings.filterwarnings("ignore")


def laplacian_score_with_target(X, y):
    """
    Compute the Laplacian score for feature selection in regression tasks.
    Arguments:
    - X: Input data (torch tensor of shape [n_samples, n_features])
    - y: Target values (torch tensor of shape [n_samples])
    Returns:
    - laplacian_scores: A tensor with Laplacian scores for each feature
    """
    n_samples, n_features = X.shape

    # Step 1: Calculate pairwise Euclidean distance between samples for feature data
    dist_matrix = pdist(X.numpy(), 'minkowski')  # Change to euclidean if not working
    dist_matrix = squareform(dist_matrix)

    # Convert the numpy arrays back to torch tensors with float32 type
    dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32)

    # Step 2: Construct the similarity matrix (Gaussian similarity)
    sigma = torch.median(dist_matrix)  # Use median distance as sigma
    similarity_matrix = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

    # Step 3: Construct the degree matrix (diagonal matrix of row sums of similarity matrix)
    degree_matrix = torch.diag(similarity_matrix.sum(dim=1))

    # Step 4: Compute the Laplacian matrix: L = D - W
    laplacian_matrix = degree_matrix - similarity_matrix

    # Step 5: Ensure X and y are the same dtype
    X = X.to(torch.float32)
    y = y.to(torch.float32)

    # Step 6: Compute the Laplacian score for each feature
    laplacian_scores = []
    for feature_idx in range(n_features):
        feature = X[:, feature_idx]

        # Step 6.1: Compute the difference between feature and the target (y)
        feature_diff = feature.unsqueeze(1) - X  # Difference between feature and all other features
        target_diff = y.unsqueeze(1) - y  # Difference between target values

        # Step 6.2: Instead of matrix multiplication, directly use the feature differences
        # Compute the weighted sum of squared differences using the Laplacian matrix
        weighted_diff = torch.matmul(laplacian_matrix, feature_diff)

        # Compute the score by considering both feature_diff and target_diff directly
        score = torch.sum(weighted_diff * target_diff, dim=1).mean()

        laplacian_scores.append(score.item())

    return torch.tensor(laplacian_scores)


'''
##############################################################################################################

Everything below is the original benchmark/comparison script (MI, HSIC, MIC, LS) and is kept
for reference only -- not executed on import.

##############################################################################################################

import pandas as pd
import numpy as np


def hsic_torch(X, Y, sigma=1.0):
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC) using PyTorch.

    Parameters:
    - X: torch.Tensor of shape (n_samples, d_x) - Feature matrix
    - Y: torch.Tensor of shape (n_samples, d_y) - Target matrix
    - sigma: float - Bandwidth for Gaussian kernel

    Returns:
    - hsic_value: float - HSIC score
    """
    n = X.shape[0]


    def gaussian_kernel(A, sigma):

      dist_matrix = pdist(A.numpy(), 'minkowski')

      dist_matrix = squareform(dist_matrix)

      #dist_matrix = torch.cdist(X, X, p=2)

      dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32)

      similarity_matrix = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

      return similarity_matrix

    K = gaussian_kernel(X, sigma)
    L = gaussian_kernel(Y, sigma)


    H = torch.eye(n) - (1 / n) * torch.ones((n, n))


    HSIC = torch.trace(K @ H @ L @ H) / (n) ** 2
    return HSIC.item()

mi_f,hsic_f,ls_f,mic_f = [],[],[],[]
for i  in range(10):

    n_samples = 100
    n_features = 1000

    torch.manual_seed(i)
    X = torch.rand(n_samples, n_features)
    y = ((X[:, 1] + torch.log(X[:, 3])) / (X[:, 5] + X[:, 6])).reshape(-1, 1)

    from sklearn.feature_selection import mutual_info_regression

    mi_scores = torch.tensor([
        mutual_info_regression(X[:, i].unsqueeze(1).numpy(), y.numpy()).item()
        for i in range(n_features)
    ])


    hsic_scores = torch.tensor([hsic_torch(X[:, i].unsqueeze(1), y) for i in range(n_features)])


    mi_hsic_scores = (mi_scores / mi_scores.mean()) + (hsic_scores / hsic_scores.mean())


    comparison_df = pd.DataFrame({
        'Feature': [f'X{i}' for i in range(n_features)],
        'MI': mi_scores.numpy(),
        'HSIC': hsic_scores.numpy(),
        'MI-HSIC': mi_hsic_scores.numpy()
    })


    laplacian_scores = laplacian_score_with_target(X, y)

    val,ind = torch.topk((laplacian_scores),50)


    comparison_df['LS'] = abs(laplacian_scores.numpy())

    comparison_df['HSIC-LS'] = comparison_df['HSIC']/comparison_df['HSIC'].mean() + comparison_df['LS']/comparison_df['LS'].mean()

    features = comparison_df.sort_values(by='MI', ascending=False).head(100).Feature.tolist()
    index = comparison_df.sort_values(by='MI', ascending=False).head(100).index.tolist()
    subset = set(['X1','X3','X5','X6'])
    features = set(features)

    if subset.issubset(features):
        print("All features are present.")
        mi_f.append(4)
    else:
        present_features = subset.intersection(features)
        print("The following features are present:", present_features)
        mi_f.append((present_features))


    features = comparison_df.sort_values(by='HSIC', ascending=False).head(100).Feature.tolist()
    index = comparison_df.sort_values(by='HSIC', ascending=False).head(100).index.tolist()


    features = set(features)

    if subset.issubset(features):
        print("All features are present.")
        hsic_f.append(4)
    else:
        present_features = subset.intersection(features)
        print("The following features are present:", present_features)
        hsic_f.append((present_features))


    features = comparison_df.sort_values(by='LS', ascending=False).head(100).Feature.tolist()
    index = comparison_df.sort_values(by='LS', ascending=False).head(100).index.tolist()

    features = set(features)

    if subset.issubset(features):
        print("All features are present.")
        ls_f.append(4)
    else:
        present_features = subset.intersection(features)
        print("The following features are present:", present_features)
        ls_f.append((present_features))



    from minepy import MINE,cstats

    mine1=[]
    for i in range(X.shape[1]):
        mine=MINE(alpha=1.0, c=40, est="mic_approx")
        mine.compute_score(X[:,i].numpy(),y.flatten().numpy())
        mine1.append(mine.mic())

    k,z = torch.topk(torch.tensor(mine1),k=100)


    comparison_df['MIC'] = mine1
    features = comparison_df.sort_values(by='MIC', ascending=False).head(100).Feature.tolist()
    index = comparison_df.sort_values(by='MIC', ascending=False).head(100).index.tolist()

    features = set(features)

    if subset.issubset(features):
        print("All features are present.")
        mic_f.append(4)
    else:
        present_features = subset.intersection(features)
        print("The following features are present:", present_features)
        mic_f.append((present_features))

    print('########################************************************#################################### \n')
# %%
'''
