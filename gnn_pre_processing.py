import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["torch-geometric", "numpy", "scikit-learn", "scipy"]

# Check and install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Required imports
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.cluster import SpectralClustering
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.io import mmwrite

# Prune matrices
def prune_dense_matrix(matrix, threshold=0.1):
    """ Prune the dense matrix by thresholding small weights. """
    pruned_matrix = np.where(abs(matrix) < threshold, 0, matrix)
    return pruned_matrix

def prune_sparse_matrix(sparse_matrix, percentage=10):
    """ Prune the sparse matrix by randomly removing a percentage of edges. """
    coo_matrix = sparse_matrix.tocoo()
    num_edges_to_prune = int(coo_matrix.nnz * percentage / 100)
    indices_to_prune = np.random.choice(coo_matrix.nnz, num_edges_to_prune, replace=False)
    mask = np.ones(coo_matrix.nnz, dtype=bool)
    mask[indices_to_prune] = False
    pruned_sparse_matrix = csr_matrix((coo_matrix.data[mask], (coo_matrix.row[mask], coo_matrix.col[mask])), shape=sparse_matrix.shape)
    return pruned_sparse_matrix

def graph_compression_to_coo(adjacency_matrix):
    """ Convert adjacency matrix to COO format, keeping only upper triangular part. """
    coo_matrix = adjacency_matrix.tocoo()
    upper_tri_mask = coo_matrix.row < coo_matrix.col
    compressed_coo_matrix = csr_matrix((coo_matrix.data[upper_tri_mask], (coo_matrix.row[upper_tri_mask], coo_matrix.col[upper_tri_mask])), shape=adjacency_matrix.shape).tocoo()
    return compressed_coo_matrix

def write_custom_mtx(filename, sparse_matrix):
    """ Write a sparse matrix to a file in a custom .mtx format, sorted by column. """
    sorted_indices = np.lexsort((sparse_matrix.row, sparse_matrix.col))
    sorted_rows = sparse_matrix.row[sorted_indices]
    sorted_cols = sparse_matrix.col[sorted_indices]
    lines = [f"{sparse_matrix.shape[0]} {sparse_matrix.shape[1]} {sparse_matrix.nnz}"]
    lines.extend([f"{row + 1} {col + 1}" for row, col in zip(sorted_rows, sorted_cols)])
    with open(filename, 'w') as f:
        f.write("\n".join(lines))

# Main execution
if __name__ == "__main__":
    # Load Cora dataset
    dataset = Planetoid(root='~/somewhere/Cora', name='Cora')
    dataset1 = Planetoid(root='~/somewhere/PubMed', name='PubMed')
    datasets = {
        'cora.mtx':dataset[0],
        'pubmed.mtx': dataset1[0]
    }

    for dataName, data in datasets.items():
      # Extract and prune matrices
      sparse_adj_matrix = to_scipy_sparse_matrix(data.edge_index)
      dense_feature_matrix = data.x.numpy()  # Convert to numpy for pruning function
      pruned_dense_matrix = prune_dense_matrix(dense_feature_matrix)
      pruned_sparse_matrix = prune_sparse_matrix(sparse_adj_matrix)

      # Compress and save matrix
      compressed_sparse_coo_matrix = graph_compression_to_coo(pruned_sparse_matrix)
      write_custom_mtx(dataName, compressed_sparse_coo_matrix)
