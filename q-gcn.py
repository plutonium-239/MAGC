import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import numpy as np
from tqdm import tqdm
import random
import os
from utils import *
import argparse
from models import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

dataset_name = 'Cora'
if 'dataset' in os.environ:
	dataset_name = os.environ['dataset']
print(dataset_name)

def get_dataset(dataset_name):
	if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
		return pyg.datasets.Planetoid, dataset_name
	if dataset_name in ['USA', 'Brazil', 'Europe']:
		return pyg.datasets.Airports, dataset_name

dataset_class, pyg_dataset_name = get_dataset(dataset_name)
dataset = dataset_class(name=pyg_dataset_name, root=f'data/')
data = dataset[0].to(device)
# print('p:', dataset[0])

edge_list = dataset[0].edge_index
e = edge_list.shape[1]  # number of edges
labels = dataset[0].y.to(device)
edge_list = edge_list.to(device)
# print("Homophilic ratio : " + str(pyg.utils.homophily(edge_list, labels, method='edge')))


adj = pyg.utils.to_dense_adj(dataset[0].edge_index)
adj = adj[0]


X = dataset[0].x
p = X.shape[0]  # Number of nodes

# X = X.to_dense() remove this
if dataset_name in ['USA', 'Brazil', 'Europe']:
	degrees = adj.sum(dim=1).long()
	features = torch.zeros(p, int(degrees.max())+1,device=device)
	features[torch.arange(p),degrees] = 1
	X = features
	data.x = features
n = X.shape[1]  # feature dimension
k = len(torch.unique(labels))  # Number of cluster and coarsened dimension

sparsity_original = 2*e/(p*(p-1))
# print("Sparsity of original graph : " + str(sparsity_original))

# print('X:', X.shape, 'adj', adj.shape)
nn = int(1*p)
# X = X[:nn, :]
# adj = adj[:nn, :nn]
# labels = labels[:nn]

theta = get_laplacian(adj)
try:
	theta = convertScipyToTensor(theta)
except:
	pass
theta = theta.to(device)
# print(f"theta: {theta.shape}")

B = get_modularity_matrix(adj)  # B -> modularity matrix
try:
	B = convertScipyToTensor(B)
except:
	pass
B = B.to(device)
# print(f"B: {B.shape}")

try:
	X = convertScipyToTensor(X)
	X = X.to_dense()
except:
	pass
X = X.to(device)

J = (torch.ones((k, k) ,device=device)/k)

def main(args):
	scaling = {
		'Cora': 1e5,
		'CiteSeer': 1e5,
		'PubMed': 1e14,
	}
	if args.loss_scaler == -1:
		if dataset_name in scaling:
			args.loss_scaler = scaling[dataset_name]
		else:
			args.loss_scaler = 1e-5
	
	if args.random_seed is not None and args.random_seed != -1:
		# random_seed = random.randint(0, 1e4)
		torch.manual_seed(args.random_seed)
		random.seed(args.random_seed)
		np.random.seed(args.random_seed)
		torch.cuda.manual_seed_all(args.random_seed)
		
	model = GCN(n, k)
	model.train()

	model.to(device)
	optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
	metrics_time = []

	for epoch in tqdm(range(args.epochs)):
		optimizer.zero_grad()
		C_soft = model(data)
		pred_clusters = C_soft.argmax(dim=1)
		C = (C_soft == C_soft.max(dim=1)[0][:,None]).float()
		
		X_tilde = torch.linalg.pinv(C) @ X # dim of X_tilde: [k,n]

		cluster_sizes = C.sum(axis=0)
	
		cluster_sizes_norm = torch.linalg.norm(cluster_sizes, ord=2) 
		
		coarsened_theta_term = -torch.logdet(C_soft.T@theta@C_soft + J)
		coarsened_features_term = torch.trace(X_tilde.T@C_soft.T@theta@C_soft@X_tilde)
		coarsening_constraint_term = (torch.norm(C_soft@X_tilde - X, p='fro')**2)/2
		C_sparsity_term = (torch.norm(C_soft.T.norm(dim=1, p=1), p=2)**2)/2
		modularity_term = -torch.trace(C_soft.T@B@C_soft)/(2*e)
		collapse_reg_term = np.sqrt(k)/p*cluster_sizes_norm - 1

		loss = (args.gamma*coarsened_theta_term + coarsened_features_term + args.alpha*coarsening_constraint_term + \
				args.lambdap*C_sparsity_term + args.beta*modularity_term + args.delta*collapse_reg_term)/args.loss_scaler

		loss.backward()
		optimizer.step()

		metrics = model_eval(adj.cpu(), pred_clusters.cpu(), labels.cpu())
		metrics['loss'] = loss.cpu().detach()
		metrics_time.append(metrics)

		# scheduler.step(loss)

		# if args.umap and epoch%25==0:
		# 	fig_labels, fig_clusters, fig_X_tilde, _ = embed_umap_plot(X, X_tilde, labels, pred_clusters)
		# 	experiment.log_figure('GT Labels', fig_labels, step=epoch)
		# 	experiment.log_figure('Clusters', fig_clusters, step=epoch)
		# 	experiment.log_figure('Coarsened Graph', fig_X_tilde, step=epoch)


	params = {
		'model': 'GCN',
		'p': p,
		'k': k,
		'n': n,
		'dataset': dataset_name,
	}
	return params, metrics, metrics_time
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
	parser.add_argument('--epochs', type=int, default=400, help="Number of Training Epochs")
	parser.add_argument('--alpha', type=float, default=2000, help="alpha weight")
	parser.add_argument('--beta', type=float, default=100, help="beta weight")
	parser.add_argument('--gamma', type=float, default=500, help="gamma weight")
	parser.add_argument('--delta', type=float, default=500, help="delta weight")
	parser.add_argument('--lambdap', type=float, default=500, help="lambda weight")
	parser.add_argument('--loss_scaler', type=float, default=-1, help="Scale the whole loss")
	parser.add_argument('--random_seed', type=int, default=-1, help="Random seed")
	parser.add_argument('--umap', action='store_true', default=False, help="Make UMAP plot")

	args = parser.parse_args()

	params, results, metrics_time = main(args)
	
	import pandas as pd
	from datetime import datetime
	df = pd.DataFrame(metrics_time, columns=list(results.keys())).round(4)

	print(results)
	fname = f'results/q-gcn-{datetime.now().strftime("%d-%m_%H_%M")}.csv'
	df.to_csv(fname)
	print('Saved at ' + fname)
