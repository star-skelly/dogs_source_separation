import double_gmm as dogs
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# inspired by the deep spectral segmentation paper

def prepare_astrophysics_features(starlet_cube):
    # starlet_cube shape: (E, C, X, Y)
    # We want (Spatial_Pixels, Feature_Dim)
    # 1. Permute to (X, Y, E, C)
    # 2. Reshape to (X*Y, E*C)
    E, C, X, Y = starlet_cube.shape
    feats = starlet_cube.permute(2, 3, 0, 1).reshape(X * Y, E * C)
    
    # Normalize features (Crucial for spectral clustering)
    feats = F.normalize(feats, p=2, dim=1)
    return feats, (X, Y)

def get_eigenvectors_astronomy(feats, K=5):
    """
    feats: (N_pixels, D_features)
    K: number of eigenvectors to compute
    """
    # 1. Compute Affinity Matrix (Cosine Similarity)
    # Using a threshold to keep the matrix sparse/clean
    A = torch.matmul(feats, feats.T)
    A = torch.clamp(A, min=0) # Keep only positive correlations
    
    # 2. Convert to Numpy for Scipy (more stable for eigsh)
    W = A.cpu().numpy()
    d = np.sum(W, axis=1)
    D = np.diag(d)
    L = D - W # Unnormalized Laplacian
    
    # 3. Solve Generalized Eigenvalue Problem: L * v = lambda * D * v
    # This gives us the Normalized Cut solution
    try:
        # We want 'SM' (Smallest Magnitude) eigenvalues
        # The first eigenvector is usually constant; we want 1 to K
        eigenvalues, eigenvectors = ssl.eigsh(L, k=K+1, which='SM', M=D)
    except Exception as e:
        print(f"Eigsh failed, falling back: {e}")
        eigenvalues, eigenvectors = np.linalg.eigh(L) # Dense fallback
        
    return torch.from_numpy(eigenvectors)

K = 5
feats, (X, Y) = prepare_astrophysics_features(dogs.cube)
eigenvectors = get_eigenvectors_astronomy(feats, K=K)

# Reshape the 2nd eigenvector back to the image grid
# index 1 is usually the first non-trivial structural component
segmentation_map = eigenvectors[:, 1].reshape(X, Y)

# Thresholding (Zero-crossing is a common heuristic)
binary_mask = segmentation_map > 0

def visualize_spectral_results(starlet_cube, eigenvectors, X, Y):
    """
    starlet_cube: (E, C, X, Y)
    eigenvectors: (N_pixels, K)
    """
    # 1. Prepare the background (sum over energy/levels for a 'white light' image)
    # We use index 0 of starlet_cube as a proxy for the raw intensity
    reference_img = starlet_cube[0, 0].cpu().numpy() 

    # 2. Setup the plot grid
    # Row 1: Input Starlet/Raw data
    # Row 2: Eigenvectors (The segmentation "features")
    fig, axes = plt.subplots(2, K - 1, figsize=(20, 10))
    fig.suptitle('Unsupervised Spectral Segmentation: Starlet Cube to Eigenvectors', fontsize=16)

    # Plot Starlet levels
    for i in range(3):
        axes[0, i].imshow(starlet_cube[0, i].cpu().numpy(), cmap='inferno')
        axes[0, i].set_title(f'Starlet Level {i}')
        axes[0, i].axis('off')

    # Plot Eigenvectors (skipping index 0 as it's the trivial constant)
    for i in range(1, K):
        # Reshape the i-th eigenvector column back to spatial dimensions
        eig_img = eigenvectors[:, i].reshape(X, Y).cpu().numpy()
        
        ax = axes[1, i-1]
        im = ax.imshow(eig_img, cmap='RdBu_r') # Red-Blue highlights positive/negative clusters
        ax.set_title(f'Eigenvector {i}')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("output/spectral_eigen_results.png")

# Usage assuming you have the outputs from the previous step:
X, Y = dogs.cube.shape[2], dogs.cube.shape[3]
visualize_spectral_results(dogs.cube, eigenvectors, X, Y)