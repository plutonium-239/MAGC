import torch
import torch.nn as nn
import torch_geometric as pyg
import numpy as np
from tqdm import tqdm
import random
import os
import scipy.sparse as sp
from utils import *
import argparse

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
# print('p:', dataset[0])

edge_list = dataset[0].edge_index
e = edge_list.shape[1]  # number of edges
labels = dataset[0].y

# print("Homophilic ratio : " + str(pyg.utils.homophily(edge_list, labels, method='edge')))


adj = pyg.utils.to_dense_adj(dataset[0].edge_index)
adj = adj[0]


X = dataset[0].x
# X = X.to_dense() remove this
p = X.shape[0]  # Number of nodes
n = X.shape[1]  # feature dimension
k = len(torch.unique(labels))  # Number of cluster and coarsened dimension

sparsity_original = 2*e/(p*(p-1))
# print("Sparsity of original graph : " + str(sparsity_original))


# print('X:', X.shape, 'adj:', adj.shape)

nn = int(1*p)
# X = X[:nn, :]
# adj = adj[:nn, :nn]
# labels = labels[:nn]

theta = get_laplacian(adj)
theta = theta.to(device)
# print(f"theta: {theta.shape}")

B = get_modularity_matrix(adj)  # B -> modularity matrix
# print(f"B: {B.shape}")

features = X.numpy()
temp = CustomDistribution(seed=1)
temp2 = temp()  # get a frozen version of the distribution
X_tilde = sp.random(k, n, density=0.25, random_state=1, data_rvs=temp2.rvs)
C = sp.random(p, k, density=0.25, random_state=1, data_rvs=temp2.rvs)

def q_fgc(args, C, X_tilde, theta, B, X):
	ones = sp.csr_matrix(np.ones((k, k)))
	ones = convertScipyToTensor(ones)
	ones = ones.to_dense()
	J = np.outer(np.ones(k), np.ones(k))/k
	J = sp.csr_matrix(J)
	J = convertScipyToTensor(J)
	J = J.to_dense()
	zeros = sp.csr_matrix(np.zeros((p, k)))
	zeros = convertScipyToTensor(zeros)
	zeros = zeros.to_dense()
	X_tilde = convertScipyToTensor(X_tilde)
	X_tilde = X_tilde.to_dense()
	C = convertScipyToTensor(C)
	C = C.to_dense()
	eye = torch.eye(k)
	try:
		theta = convertScipyToTensor(theta)
	except:
		pass
	try:
		B = convertScipyToTensor(B)
	except:
		pass
	try:
		X = convertScipyToTensor(X)
		X = X.to_dense()
	except:
		pass

	X_tilde = X_tilde.to(device)
	C = C.to(device)
	B = B.to(device)
	X = X.to(device)
	J = J.to(device)
	zeros = zeros.to(device)
	ones = ones.to(device)
	eye = eye.to(device)
	print('C:', C.shape)

	metrics_time = []
	log_vals_time = []

	def update(X_tilde, C, i):
		global L
		thetaC = theta@C
		CT = torch.transpose(C, 0, 1)
		X_tildeT = torch.transpose(X_tilde, 0, 1)
		CX_tilde = C@X_tilde
		t1 = CT@thetaC + J
		if torch.det(t1) == 0:
			idx = torch.where(torch.diag(t1)==0)[0]
			t1[idx, idx] = 1/k
		term_bracket = torch.linalg.pinv(t1)
		thetacX_tilde = thetaC@(X_tilde)
		cluster_sizes = C.sum(axis=0)
		cluster_sizes_norm = torch.linalg.norm(cluster_sizes, ord=2) # Not using vector_norm

		L = 1/k

		log_vals = {}

		t1 = -2*args.gamma*(thetaC@term_bracket)
		t2 = args.alpha*(CX_tilde-X)@(X_tildeT)
		t3 = 2*thetacX_tilde@(X_tildeT)
		t4 = args.lambdap*(C@ones)
		t5 = -args.beta/2 * ((B.T+B)@C)  # Modularity
		# collapse regularization
		t6 = args.delta*np.sqrt(k)/p*(cluster_sizes/cluster_sizes_norm)

		T2 = (t1+t2+t3+t4+t5+t6)/L
		Cnew = (C-T2).maximum(zeros)
		t1 = CT@thetaC*(2/args.alpha)
		t2 = CT@C
		t1 = torch.linalg.pinv(t1+t2)
		t1 = t1@CT
		t1 = t1@X
		X_tilde_new = t1
		Cnew[Cnew < args.thresh] = args.thresh
		for i in range(len(Cnew)):
			Cnew[i] = Cnew[i]/torch.linalg.norm(Cnew[i], 1)
		for i in range(len(X_tilde_new)):
			X_tilde_new[i] = X_tilde_new[i] / \
				torch.linalg.norm(X_tilde_new[i], 1)
		return X_tilde_new, Cnew, log_vals

	for i in tqdm(range(args.update_iters)):
		X_tilde, C, log_vals = update(X_tilde, C, i)
		pred_clusters = C.argmax(axis=1)
		metrics_time.append(model_eval(adj.cpu(), pred_clusters.cpu(), labels.cpu()))
		log_vals_time.append(log_vals)
		# Implementing Early Stopping
		if i > 39:
			last_5 = metrics_time[-5:]
			flag_continue = False
			for j in range(len(last_5) - 1):
				diff = abs(last_5[j+1]['NMI'] - last_5[j]['NMI'])
				# print(f"Modularity at {j+1}: {last_5[j+1]['Modularity']}" )
				# print(f"Modularity at {j}: {last_5[j]['Modularity']}" )
				if abs(diff/last_5[j]['NMI']) > 0.01:
					flag_continue = True
			if not flag_continue:
				print(f"Early Stopping at {i}, no increase > {0.01*last_5[j]['NMI']:.4f} for 5 iters.")
				break
	return X_tilde, C, metrics_time, log_vals_time


def main(args):
	if args.random_seed == -1:
		args.random_seed = random.randint(0, 1e5)
	X_tilde = sp.random(k, n, density=0.15,random_state=1, data_rvs=temp2.rvs)
	C = sp.random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
	X_t_0, C_0, metrics_time, log_vals_time = q_fgc(args, C, X_tilde, theta, B, X)
	
	getSparsityAndHomophily(C_0, theta)

	pred_clusters = C_0.argmax(axis=1)
	params = {
		'p': p,
		'k': k,
		'n': n,
		'dataset': dataset_name
	}
	return params, model_eval(adj.cpu(), pred_clusters.cpu(), labels.cpu()), metrics_time


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--alpha', type=float, default=1000.0, help="alpha weight")
	parser.add_argument('--beta', type=float, default=100, help="beta weight")
	parser.add_argument('--gamma', type=float, default=1000, help="gamma weight")
	parser.add_argument('--delta', type=float, default=10, help="delta weight")
	parser.add_argument('--lambdap', type=float, default=100, help="lambda weight")
	parser.add_argument('--thresh', type=float, default=1e-10, help="learning rate")
	parser.add_argument('--update_iters', type=int, default=40, help="number of update iterations")
	parser.add_argument('--random_seed', type=int, default=-1, help="Random seed")

	args = parser.parse_args()

	params, results, metrics_time = main(args)

	import pandas as pd
	from datetime import datetime
	df = pd.DataFrame(metrics_time, columns=list(results.keys())).round(4)

	print(results)
	fname = f'results/q-fgc-{datetime.now().strftime("%d-%m_%H-%M")}.csv'
	df.to_csv(fname)
	print('Saved at ' + fname)