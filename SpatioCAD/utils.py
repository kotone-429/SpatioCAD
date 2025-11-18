import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy
import scanpy as sc
from scipy.sparse import issparse,eye,csc_matrix
from numpy.linalg import norm
from typing import Optional
from anndata import AnnData

def compute_laplacian(adjacency_matrix):
    """
    Compute the standard laplacian matrix.

    Args:
        adjacency_matrix (scipy.sparse.csc_matrix): 
            The adjacency matrix constructed from spatial coordinates.

    Returns:
        L (scipy.sparse.csc_matrix):
            The standard laplacian matrix.
    """

    # Calculate the degree matrix
    degree_vector = np.array(adjacency_matrix.sum(axis=1)).flatten()
    degree_matrix = sp.diags(degree_vector, format='csc') 
    
    # Calculate the standard laplacian matrix
    if adjacency_matrix.format != 'csc':
        adjacency_matrix_processed = adjacency_matrix.tocsc()
    else:
        adjacency_matrix_processed = adjacency_matrix
        
    L = degree_matrix - adjacency_matrix_processed

    return L

def compute_node_attributed_laplacian(adjacency_matrix, cell_counts_vector):
    """
    Compute the node attributed laplacian matrix.

    Args:
        adjacency_matrix (scipy.sparse.csc_matrix): 
            The adjacency matrix constructed from spatial coordinates.
        cell_counts_vector (np.ndarray):
            The cell density at each spatial location.
 
    Returns:
        L_new (scipy.sparse.csc_matrix):
            The node attributed laplacian matrix.
    """
    if not sp.issparse(adjacency_matrix):
        adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        
    # Calculate the node attributed degree matrix
    d_loss_vec = adjacency_matrix @ cell_counts_vector
    D_loss = sp.diags(d_loss_vec, format='csc')
    
    # Calculate the node attributed laplacian matrix
    n_sqrt = np.sqrt(cell_counts_vector)
    N_diag_sqrt = sp.diags(n_sqrt, format='csc')
    L_new = D_loss - (N_diag_sqrt @ adjacency_matrix.tocsc() @ N_diag_sqrt)
    
    return L_new

def compute_diff_time(expression_matrix, eigenvalues, eigenvectors, cutoff_error: float = 10.0):
    """
    Compute characteristic diffusion time.

    Args:
        expression_matrix (scipy.sparse.csc_matrix): 
            The processed gene expression matrix.
        eigenvalues (np.ndarray):
            The eigenvalues of standard or node attributed laplacian matrix.
        eigenvectors (np.ndarray):
            The eigenvectors of standard or node attributed laplacian matrix.
        cutoff_error (float, optional):
            A parameter controlling the precision of the diffusion time calculation.
            The diffusion time is determined when the diffusion change rate drops below
            a threshold of e^(-cutoff_error). Defaults to 10.0.
 
    Returns:
        diffusion_times (np.ndarray):
            The diffusion time of each gene.
    """
    thete_matrix = eigenvectors.T @ expression_matrix
    eigenvalues_matrix = eigenvalues[:, np.newaxis]
    log_abs_c = np.log(np.abs(eigenvalues_matrix *thete_matrix) + 1e-12)
    
    time_matrix = (log_abs_c + cutoff_error) / eigenvalues_matrix
    diffusion_times = np.max(time_matrix, axis=0)
    return diffusion_times

def compute_cell_counts(expression_matrix):
    """
    Compute the cell density at each spatial location.

    Args:
        expression_matrix (scipy.sparse.csc_matrix): 
            The processed gene expression matrix.

    Returns:
        cell_counts (np.ndarray):
            The cell density at each spatial location.
    """
    cell_counts_data = np.array(expression_matrix.sum(axis=1)).flatten()
    cell_counts_data[cell_counts_data > np.percentile(cell_counts_data, 99)] = np.percentile(cell_counts_data,99) 
    cell_counts_data[cell_counts_data < np.percentile(cell_counts_data, 1)] = np.percentile(cell_counts_data,1) 
    cell_counts = cell_counts_data / min(cell_counts_data)
    return cell_counts

def compute_eigens(L, num_eigenvalues: int = 20):
    """
    Compute the num_eigenvalues smallest non-zero eigenvalues and 
    their corresponding eigenvectors for the given laplacian matrix.

    Args:
        L (scipy.sparse.csc_matrix): 
            The standard or node attributed laplacian matrix.
        num_eigenvalues (int, optional):
            The number of eigenmodes used to approximate the diffusion process.
            Defaults to 20.

    Returns:
        eigenvalues (np.ndarray):
            The eigenvalues of the given laplacian matrix.
        eigenvectors (np.ndarray):
            The eigenvectors of the given laplacian matrix.
    """
    eigenvalues, eigenvectors = eigsh(L, k=num_eigenvalues+1, which='SM')
    # Exclude the 0 eigenvalue
    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]
    return eigenvalues, eigenvectors

def power_transform_adata(adata, p: float = 2):
    """
    Apply p-th power transformations to the data.
    The transformation is applied to `adata.X`.

    Args:
        adata (anndata.AnnData): 
            The data matrix to transform.
        p (float, optional):
            The exponent (power) to apply in the transformation. Defaults to 2.0.

    Returns:
        None
    """
    if p < 1:
        raise ValueError("Power 'p' must be greater than or equal to 1.")
    if p == 1:
        return adata

    if issparse(adata.X):
        adata.X = adata.X.power(p)
    else:
        adata.X = np.power(adata.X, p)

    return adata