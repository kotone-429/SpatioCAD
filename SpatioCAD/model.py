import numpy as np
import scipy.sparse as sp
import scanpy as sc
import pandas as pd
from typing import Optional
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from .utils import compute_laplacian, compute_diff_time, compute_cell_counts,compute_eigens,power_transform_adata,compute_node_attributed_laplacian

class SpatioCAD:
    def __init__(self, adata: Optional[AnnData] = None):
        self.adata = None
        self.adjacency_matrix = None

    def read_h5ad(self, file):
        """
        Read an h5ad file and sets the 'self.adata' attribute with the loaded data.

        Args:
            file (str): 
                Path to the h5ad file to be read.

        Returns:
            None
        """
        self.adata = sc.read(file)
        self.adata.layers['counts'] = self.adata.X.copy()

    def preprocess(
            self,
            min_cells: int = 100,
            min_genes: int = 1,
            gene_normalize: bool = True,
            log1p: bool = True,
    ):
        sc.pp.filter_genes(self.adata, min_cells=min_cells)
        sc.pp.filter_cells(self.adata, min_genes=min_genes)
        if gene_normalize:
            adata_T = self.adata.T
            sc.pp.normalize_total(adata_T, target_sum=1e4, inplace=True)
            self.adata = adata_T.T
        if  log1p:
            sc.pp.log1p(self.adata)

    def compute_adjacency_matrix(
            self,
            k_times: float = 1.0,
            local_scale_k_times: float = 0.5
    ):  
        """
        Compute a weighted adjacency matrix based on Mutual Nearest Neighbors (MNN)
        using a Gaussian kernel with adaptive local scales (sigmas).
        The results are stored in 'self.adjacency_matrix' attribute,
        which is a scipy.sparse.csr_matrix.

        Args:
            k_times (float, optional): 
                Multiplier for sqrt(N) to determine the number of neighbors (k) 
                for the MNN graph construction. Defaults to 1.0.

            local_scale_k_times (float, optional): 
                Multiplier for log(N) to determine the k-th neighbor 
                used for calculating the local scale (sigma). Defaults to 0.5.

        Returns:
            None
        """
        coords = self.adata.obsm['spatial']
        n_samples = coords.shape[0]

        k = int(np.round(k_times*np.sqrt(n_samples)))
        local_scale_k = int(np.round(local_scale_k_times*np.log(n_samples)))
        
        # Perform the k-NN search
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean').fit(coords)
        distances_all, indices_all = nbrs.kneighbors(coords)

        # Compute the Mutual Nearest Neighbors (MNN) graph
        neighbor_indices = indices_all[:, 1:k+1]
        neighbor_dists = distances_all[:, 1:k+1]

        rows = np.repeat(np.arange(n_samples), neighbor_indices.shape[1])
        cols = neighbor_indices.flatten()
        dists_flat = neighbor_dists.flatten()

        A_dist = sp.csr_matrix((dists_flat, (rows, cols)), shape=(n_samples, n_samples))
        MNN_structure = A_dist.multiply(A_dist.T > 0)

        # Calculate the Gaussian kernel weights for each MNN edge
        mnn_rows, mnn_cols = MNN_structure.nonzero()

        dist_ij = np.asarray(MNN_structure.data).flatten()
        sigmas_i = distances_all[:, local_scale_k]
        sigmas_i[sigmas_i == 0] = 1e-5

        sigma_product = sigmas_i[mnn_rows] * sigmas_i[mnn_cols]
        weights = np.exp(-dist_ij**2 / sigma_product)

        # Build the final weighted adjacency matrix
        self.adjacency_matrix = sp.csr_matrix((weights, (mnn_rows, mnn_cols)), shape=(n_samples, n_samples))
        
    def filter_non_pattern_genes(
        self,
        n_components_range: range = range(1, 10),
        n_noise_components: int = 1,
        probability_threshold: float = 0.95
    ):
        """
        Compute Roughness Score (RS) for each gene to quantify the signal change 
        during the initial phase of diffusion. This score is then used to 
        filter noise genes via Gaussian Mixture Model (GMM).
        The optimal number of GMM components is determined by BIC.
        Among them, the n_noise_components GMM components 
        with the highest mean values are regarded as noise components.
        The resulting RS are stored in 'self.adata.var['roughness_scores']' attribute.
        The resulting gene labels are stored in 'self.adata.var['is_noise_gene']' attribute.
        The analysis results are stored in 'self.adata.uns['gmm_filter_results']' attribute.

        Args:
            n_components_range (range, optional):
                The range for testing the number of components in the GMM.
                Defaults to range(1, 10).
            n_noise_components (int, optional):
                The number of noise components. Defaults to 1.
            probability_threshold (float, optional):
                The smallest posterior probability of a gene 
                belonging to the noise components. Defaults to 0.95.
        
        Returns:
            None
        """
        if self.adjacency_matrix is None:
            raise ValueError("Error: The adjacency_matrix is not found in SpatioCAD. Please run 'compute_adjacency_matrix' first.")
        
        # Normalize gene expression
        n_cells_values = self.adata.var['n_cells'].values.astype(float)
        inv_n_cells = 1.0 / n_cells_values
        D_norm = sp.diags(inv_n_cells, format='csc')
        expression_matrix = (self.adata.X > 0).astype(np.int8)
        expression_matrix_norm = expression_matrix @ D_norm

        # Compute Roughness Score (RS)
        L = compute_laplacian(self.adjacency_matrix)
        LX_norm = L @ expression_matrix_norm
        sum_sq = LX_norm.power(2).sum(axis=0)
        roughness_scores = np.sqrt(np.asarray(sum_sq).ravel())
        self.adata.var['roughness_scores'] = roughness_scores

        # Determine the optimal number of GMM components via BIC
        scores_reshaped = roughness_scores.reshape(-1, 1)

        bics = []
        for n in n_components_range:
            gmm_test = GaussianMixture(n_components=n, random_state=42, n_init=10)
            gmm_test.fit(scores_reshaped)
            bics.append(gmm_test.bic(scores_reshaped))
        
        optimal_n = n_components_range[np.argmin(bics)]

        # Fit the distribution of RS using the optimal number of GMM components
        gmm_final = GaussianMixture(n_components=optimal_n, random_state=42, n_init=10)
        gmm_final.fit(scores_reshaped)

        # Determine the signal and noise components
        means = gmm_final.means_.flatten()
        sorted_indices = np.argsort(means)
        raw_noise_indices = sorted_indices[-n_noise_components:]
        raw_signal_indices = sorted_indices[:-n_noise_components]
        mean_order = list(raw_signal_indices) + list(raw_noise_indices)

        # Label the components
        remapping_dict = {}
        component_labels = []
        signal_counter = 1 
        noise_counter = 1
        for raw_idx in mean_order:
            if raw_idx in raw_noise_indices:
                label = f'Noise Component {noise_counter}'
                noise_counter +=1
            else:
                label = f'Signal Component {signal_counter}'
                signal_counter += 1
            remapping_dict[raw_idx] = label
            component_labels.append(label)
        
        raw_gene_components = gmm_final.predict(scores_reshaped)
        final_gene_labels = pd.Series(raw_gene_components).map(remapping_dict).values

        # Determine the signal and noise genes
        self.adata.var['gmm_component_label'] = final_gene_labels
        posterior_probs = gmm_final.predict_proba(scores_reshaped)
        prob_is_noise = posterior_probs[:, raw_noise_indices].sum(axis=1)
        is_noise_gene_mask = prob_is_noise > probability_threshold
        self.adata.var['is_noise_gene'] = is_noise_gene_mask
        noise_gene_count = np.sum(is_noise_gene_mask)
        new_noise_indices = list(range(optimal_n - n_noise_components, optimal_n))

        # Record the analysis results
        gmm_filter_results = {
            'bic_scores': bics, 
            'tested_n_components': list(n_components_range),
            'optimal_n': optimal_n,
            'component_labels': component_labels,
            'final_gmm_means': gmm_final.means_[mean_order],
            'final_gmm_covariances': gmm_final.covariances_[mean_order],
            'final_gmm_weights': gmm_final.weights_[mean_order],
            'noise_component_indices': new_noise_indices, 
            'noise_gene_count': noise_gene_count
        }

        self.adata.var['gmm_component_label'].value_counts()
        self.adata.uns['gmm_filter_results'] = gmm_filter_results

    def compute_SVG(
        self,
        power_values: list = [1, 1.5, 2],
        num_eigen : int = 50,
        cutoff_error: float = 10.0,
    ):
        """
        Identify spatially high variable genes using a diffusion-based model.
        This method calculates a characteristic diffusion time for each gene,
        which quantifies how far is its expression pattern from the stable state
        under Node-Attributed Graph Diffusion framework.
        The resulting ranks are stored in 'self.adata.var['robust_rank']' attribute.

        Args:
            power_values (list, optional):
                The series of p-th power transformations to apply to the expression
                matrix. Defaults to [1, 1.5, 2].
            num_eigen (int, optional):
                The number of eigenmodes used to approximate the diffusion process.
                Defaults to 50.
            cutoff_error (float, optional):
                A parameter controlling the precision of the diffusion time calculation.
                The diffusion time is determined when the diffusion change rate drops below
                a threshold of e^(-cutoff_error). Defaults to 10.0.
        
        Returns:
            final_df (DataFrame):
                The diffusion time results of the series of p-th power transformations and
                the final ranks.
        """
        non_noise_mask = (~self.adata.var['is_noise_gene'] 
                if 'is_noise_gene' in self.adata.var.columns 
                else slice(None))
        adata_for_svg = self.adata[:,non_noise_mask].copy()

        all_gene_diffusion_times = {}
        all_gene_rankings = {}

        for p in power_values:
            # Perform the p-th power transformations
            adata_p = power_transform_adata(adata_for_svg.copy(), p=p)
            adata_T = adata_p.T
            sc.pp.normalize_total(adata_T, target_sum=1, inplace=True)
            adata_p = adata_T.T
            expression_matrix = sp.csr_matrix(adata_p.X) if not issparse(adata_p.X) else adata_p.X.copy()

            # Calculate cell densities
            cell_counts = compute_cell_counts(expression_matrix)

            # Transform gene expression
            cell_counts_sqrt_inv = 1.0 / np.sqrt(cell_counts)
            N_sqrt_inv_diag = sp.diags(cell_counts_sqrt_inv, format='csc')
            processed_expr_cell_counts = N_sqrt_inv_diag @ expression_matrix

            # Calculate Node-Attributed adjacent matrix
            coo = self.adjacency_matrix.tocoo()
            rows, cols, data = coo.row, coo.col, coo.data
            n_i = cell_counts[rows]
            n_j = cell_counts[cols]

            new_data = data * (n_i + n_j) / (2 * n_i * n_j)
            new_adj = sp.coo_matrix((new_data, (rows, cols)), shape=self.adjacency_matrix.shape)

            # Calculate Node-Attributed laplacian
            laplacian = compute_node_attributed_laplacian(new_adj, cell_counts)
            eigenvalues, eigenvectors = compute_eigens(laplacian, num_eigenvalues=num_eigen)

            # Compute and record diffusion time
            diff_times = compute_diff_time(processed_expr_cell_counts, eigenvalues, eigenvectors, cutoff_error=cutoff_error)

            key = f'{p}_power'
            all_gene_diffusion_times[key] = diff_times
            series = pd.Series(diff_times, index=adata_p.var_names).dropna()
            all_gene_rankings[key] = series.rank(method='first', ascending=False)

        # Record the results of non-noise genes
        df_times = pd.DataFrame(all_gene_diffusion_times, index=adata_for_svg.var_names)
        df_ranks = pd.DataFrame(all_gene_rankings)

        # Calculate the final rank of non-noise genes using lexicographical sort
        lex_sort_key = df_ranks.apply(lambda row: tuple(sorted(row.dropna().astype(int))), axis=1)

        final_df_non_noise = df_times.copy()
        final_df_non_noise['lex_sort_key'] = lex_sort_key

        final_df_non_noise.sort_values(by='lex_sort_key', ascending=True, inplace=True)
        final_df_non_noise['robust_rank'] = range(1, len(final_df_non_noise) + 1)

        # Add the filtered noise genes to the final results
        final_df = final_df_non_noise.reindex(self.adata.var_names)
        final_df['is_noise_gene'] = self.adata.var['is_noise_gene']
        max_rank_for_non_noise = final_df_non_noise['robust_rank'].max()
        final_df['robust_rank'] = final_df['robust_rank'].fillna(max_rank_for_non_noise + 1)
        final_df['robust_rank'] = final_df['robust_rank'].astype(int)

        # Record the results of all genes
        final_df.sort_values(by='robust_rank', inplace=True)
        final_df.reset_index(inplace=True)
        final_df.rename(columns={'index': 'gene'}, inplace=True)

        front_columns = ['gene', 'robust_rank', 'is_noise_gene']
        other_columns = [col for col in final_df.columns if col not in front_columns]
        new_order = front_columns + other_columns
        final_df = final_df[new_order]

        rank_series_for_assignment = pd.Series(
            data=final_df['robust_rank'].values,  
            index=final_df['gene']                
        )

        self.adata.var['robust_rank'] = rank_series_for_assignment
        
        return final_df
    
    def cluster_gene(
            self,
            n_clusters: int = 6,
            n_top_genes: int = 2000,
    ):  
        """
        Cluster SVGs based on the spatial diffusion profiles.
        The resulting cluster labels are stored in 'self.adata.var['gene_cluster']' attribute.

        Args:
            n_clusters (int, optional):
                The number of clusters for SVGs. Defaults to 6.
            n_top_genes (int, optional):
                The number of SVGs used for analysis. Defaults to 2000.
        
        Returns:
            None
        """   
        # Process gene expression for clustering
        adata_cluster = self.adata.copy()
        adata_T = adata_cluster.T
        sc.pp.normalize_total(adata_T, target_sum=1, inplace=True)
        adata_cluster = adata_T.T
        expression_matrix = sp.csr_matrix(adata_cluster.X) if not issparse(adata_cluster.X) else adata_cluster.X.copy()
        cell_counts = compute_cell_counts(expression_matrix)

        # Transform gene expression
        cell_counts_sqrt_inv = 1.0 / np.sqrt(cell_counts)
        N_sqrt_inv_diag = sp.diags(cell_counts_sqrt_inv, format='csc')
        processed_expr_cell_counts = N_sqrt_inv_diag @ expression_matrix

        # Calculate Node-Attributed adjacent matrix
        coo = self.adjacency_matrix.tocoo()
        rows, cols, data = coo.row, coo.col, coo.data
        n_i = cell_counts[rows]
        n_j = cell_counts[cols]

        new_data = data * (n_i + n_j) / (2 * n_i * n_j)
        new_adj = sp.coo_matrix((new_data, (rows, cols)), shape=self.adjacency_matrix.shape)
        
        # Calculate Node-Attributed laplacian 
        laplacian = compute_node_attributed_laplacian(new_adj, cell_counts)
        eigenvalues, eigenvectors = compute_eigens(laplacian, num_eigenvalues=100)

        # Calculate the spatial diffusion profiles
        result_matrix = eigenvectors.T @ processed_expr_cell_counts
        top_indices = self.adata.var.nsmallest(n_top_genes, 'robust_rank').index
        integer_indices = [self.adata.var.index.get_loc(idx) for idx in top_indices]
        result_matrix_subset = result_matrix[:, integer_indices]
        data_points = result_matrix_subset.T

        # Cluster SVGs based on the spatial diffusion profiles and record
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        column_labels = kmeans.fit_predict(data_points)
        self.adata.var['gene_cluster'] = -1
        self.adata.var.loc[top_indices, 'gene_cluster'] = column_labels
        
    def compute_cluster_patterns(self, vote_rate: float =0.2):
        """
        Calculate the spatial pattern for each cluster.
        The resulting patterns are stored in 'self.adata.obs[f'cluster_{i}_pattern']' attribute.

        Args:
            vote_rate (float, optional):
                The cutoff of the expression ratio for each spatial location. Defaults to 0.2.
        
        Returns:
            None
        """
        # Load the expression data
        source_matrix = self.adata.layers['counts']
        if issparse(source_matrix):
            source_matrix.data = np.log1p(source_matrix.data)
        else:
            source_matrix = np.log1p(source_matrix)

        # Load the gene cluster labels
        valid_mask = self.adata.var['gene_cluster'] != -1
        if not np.any(valid_mask):
            print("Gene clusters not found. Please run cluster_gene() first.")
            return      
        filtered_adata = self.adata[:, valid_mask].copy()
        source_matrix_filtered = source_matrix[:, valid_mask]
        clusters = sorted(filtered_adata.var['gene_cluster'].unique())
        
        # Calculate the spatial pattern for each cluster
        for i in clusters:
            cluster_mask = filtered_adata.var['gene_cluster'] == i
            cluster_data = source_matrix_filtered[:, cluster_mask]
            counts = cluster_mask.sum()

            if issparse(cluster_data):
                non_zero_counts = cluster_data.getnnz(axis=1)
                sums = cluster_data.sum(axis=1).A1
                means = np.where(non_zero_counts > 0, sums / non_zero_counts, 0)
            else:
                non_zero_counts = np.count_nonzero(cluster_data, axis=1)
                means = np.true_divide(cluster_data.sum(axis=1), non_zero_counts, where=non_zero_counts>0)

            pattern = np.where(non_zero_counts >= vote_rate * counts, means, 0)
            self.adata.obs[f'cluster_{i}_pattern'] = pattern