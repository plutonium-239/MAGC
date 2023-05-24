import torch
import torch.nn.functional as F
import torch_geometric as pyg 
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse,homophily
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rv_continuous
import sklearn.metrics
import scipy.sparse as sp
import os


def get_laplacian(adj):
	b = torch.ones(adj.shape[0])
	return torch.diag(adj@b)-adj


def get_modularity_matrix(adj):
	degrees = adj.sum(axis=0)
	twice_e = degrees.sum()
	# B = A = dd^T /2e
	return adj - 1/twice_e * degrees[:, None]@degrees[:, None].T


def convertScipyToTensor(coo):
	try:
		coo = coo.tocoo()
	except:
		coo = coo
	values = coo.data
	indices = np.vstack((coo.row, coo.col))

	i = torch.LongTensor(indices)
	v = torch.FloatTensor(values)
	shape = coo.shape

	return torch.sparse.FloatTensor(i, v, torch.Size(shape))


class CustomDistribution(rv_continuous):
	def _rvs(self,  size=None, random_state=None):
		return random_state.standard_normal(size)


def getSparsityAndHomophily(C,theta):
	theta = C.T@theta@C
	adjtemp = -theta
	for i in range(adjtemp.shape[0]):
		adjtemp[i,i]=0
	adjtemp[adjtemp<0.01]=0
	temp = dense_to_sparse(adjtemp)
	edge_list_temp = temp[0]
	# ytemp = temp[1]
	# P = torch.linalg.pinv(C)
	# labels = 
	# # print(edge_list)
	number_of_edges = edge_list_temp.shape[1]
	# n = adjtemp.shape[0]
	# print("Homophilic ratio : " + str(homophily(edge_list_temp,ytemp,method='node')))
	number_of_nodes = theta.shape[0]
	sparsity = number_of_edges/(number_of_nodes*(number_of_nodes-1))
	return sparsity
	# print("Sparsity : " + str(sparsity))


# Metrics
def modularity(adjacency, clusters):
	degrees = adjacency.sum(axis=0).flatten()
	twice_e = degrees.sum()
	result = 0
	for cluster_id in np.unique(clusters):
		cluster_indices = np.where(clusters == cluster_id)[0]
		adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
		degrees_submatrix = degrees[cluster_indices]
		result += adj_submatrix.sum() - (degrees_submatrix.sum()**2) / twice_e
	return result / twice_e


def conductance(adjacency, clusters):
	inter = 0  # Number of inter-cluster edges.
	intra = 0  # Number of intra-cluster edges.fn
	cluster_indices = np.zeros(adjacency.shape[0], dtype=bool)
	for cluster_id in np.unique(clusters):
		cluster_indices[:] = 0
		cluster_indices[np.where(clusters == cluster_id)[0]] = 1
		adj_submatrix = adjacency[cluster_indices, :]
		inter += adj_submatrix[:, cluster_indices].sum()
		intra += adj_submatrix[:, ~cluster_indices].sum()
	return intra / (inter + intra)

def model_eval(adjacency, clusters, labels):
	accuracy, precision, recall = contingency_metrics(clusters, labels)
	return {	
		'Conductance': conductance(adjacency, clusters),
		'Modularity': modularity(adjacency, clusters),
		'NMI': sklearn.metrics.normalized_mutual_info_score(labels, clusters),
		'Precision': precision,
		'Recall': recall,
		'F1': 2 * precision * recall / (precision + recall),
		'Accuracy': accuracy,
		'ARI': sklearn.metrics.adjusted_rand_score(labels, clusters),
		'Clustering Accuracy': clustering_accuracy(labels, clusters)
	}


def clustering_accuracy(clusters, labels):
	cm = sklearn.metrics.confusion_matrix(labels, clusters)
	indexes = linear_sum_assignment(cm, maximize=True)
	cm2 = cm[:, indexes[1]]
	return np.trace(cm2) / np.sum(cm2)


def contingency_metrics(clusters, labels):
	contingency = sklearn.metrics.cluster.contingency_matrix(labels, clusters)
	same_class_true = np.max(contingency, 1)
	same_class_pred = np.max(contingency, 0)
	diff_class_true = contingency.sum(axis=1) - same_class_true
	diff_class_pred = contingency.sum(axis=0) - same_class_pred
	total = contingency.sum()

	tp = (same_class_true * (same_class_true - 1)).sum()
	fp = (diff_class_true * same_class_true * 2).sum()
	fn = (diff_class_pred * same_class_pred * 2).sum()
	tn = total*(total - 1) - tp - fp - fn

	accuracy = (tp + tn)/ (tp + fp + fn + tn)
	precision = tp/ (tp + fp)
	recall = tp/ (tp + fn)
	return accuracy, precision, recall

def embed_umap_plot(X, X_tilde, y, pred_clusters, latent=None, **kwargs):
	from cuml.manifold.umap import UMAP
	import plotly.express as px
	if 'n_components' in kwargs and kwargs['n_components'] == 3:
		trained_UMAP = UMAP(n_components = 3, n_neighbors = 100, verbose = 1, **kwargs).fit(X)
		X_embedded = trained_UMAP.transform(X)
		fig_labels = px.scatter_3d(X_embedded.get(), x=0, y=1, z=2, color=y.cpu())
		fig_clusters = px.scatter_3d(X_embedded.get(), x=0, y=1, z=2, color=pred_clusters.cpu())
		fig_X_tilde = px.scatter_3d(trained_UMAP.transform(X_tilde).get(), x=0, y=1, z=2, color=np.arange(X_tilde.shape[0]))
	else:
		fig_latent = None
		if latent is not None:
			latent_UMAP = UMAP(n_neighbors = 100, verbose = 2, **kwargs).fit(latent.detach())
			Z_emb = latent_UMAP.transform(latent.detach())
			fig_latent = px.scatter(Z_emb.get(), x=0, y=1, color=pred_clusters.cpu()) 

		trained_UMAP = UMAP(n_neighbors = 100, verbose = 2, **kwargs).fit(X)
		X_embedded = trained_UMAP.transform(X)
		fig_labels = px.scatter(X_embedded.get(), x=0, y=1, color=y.cpu())
		fig_clusters = px.scatter(X_embedded.get(), x=0, y=1, color=pred_clusters.cpu())
		fig_X_tilde = px.scatter(trained_UMAP.transform(X_tilde).get(), x=0, y=1, color=np.arange(X_tilde.shape[0]))
	# fig.write_image(f'images/{run_name}')
	return fig_labels, fig_clusters, fig_X_tilde, fig_latent


def acc(y_true, y_pred):
	# SAME AS clustering_accuracy above
	"""
	Calculate clustering accuracy. Require scikit-learn installed

	# Arguments
		y: true labels, numpy.array with shape `(n_samples,)`
		y_pred: predicted labels, numpy.array with shape `(n_samples,)`

	# Return
	   accuracy, in [0,1]
	"""
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.max(), y_true.max()) + 1
	w = np.zeros((D, D), dtype=np.int64)
	for i in range(y_pred.size):
		w[y_pred[i], y_true[i]] += 1
	row_ind, col_ind = linear_sum_assignment(w.max() - w)
	return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def normalize_adj(adj):
	adj = sp.coo_matrix(adj)
	adj_ = adj + sp.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
	adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	return sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):
	if not sp.isspmatrix_coo(sparse_mx):
		sparse_mx = sparse_mx.tocoo()
	coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
	values = sparse_mx.data
	shape = sparse_mx.shape
	return coords, values, shape
