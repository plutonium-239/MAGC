import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse,homophily
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rv_continuous
import sklearn.metrics
import os
import utils
from tqdm import tqdm

# GNN Architecture
class GCN(torch.nn.Module):
	def __init__(self, n, k):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(n, 128)
		self.conv2 = GCNConv(128, 64)
		self.conv3 = GCNConv(64, k)

	def reset_parameters(self):
		self.conv1.reset_parameters()
		self.conv2.reset_parameters()

	def forward(self, data):
		x, edge_index = data.x, data.edge_index

		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, p=0.1, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		x = self.conv3(x, edge_index)
		return F.softmax(x, dim=1)

class VariationalGCNEncoder(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels):
		super(VariationalGCNEncoder, self).__init__()
		self.gcn_shared = GCNConv(in_channels, hidden_channels)
		self.gcn_mu = GCNConv(hidden_channels, out_channels)
		self.gcn_logvar = GCNConv(hidden_channels, out_channels)

	def forward(self, x, edge_index):
		x = F.relu(self.gcn_shared(x, edge_index))
		mu = self.gcn_mu(x, edge_index)
		logvar = self.gcn_logvar(x, edge_index)
		return mu, logvar


def random_uniform_init(input_dim, output_dim):
	init_range = np.sqrt(6.0 / (input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return torch.nn.Parameter(initial)

class GraphConvSparse(torch.nn.Module):
	def __init__(self, input_dim, output_dim, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = random_uniform_init(input_dim, output_dim) 
		
	def forward(self, inputs, adj):
		return torch.mm(adj, torch.mm(inputs, self.weight))

class GMMVariationalGCNEncoder(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, n_clusters):
		super(GMMVariationalGCNEncoder, self).__init__()
		self.n_clusters = n_clusters

		self.gcn_shared = GraphConvSparse(in_channels, hidden_channels)
		self.gcn_mu = GraphConvSparse(hidden_channels, out_channels)
		self.gcn_logvar = GraphConvSparse(hidden_channels, out_channels)

		# GMM training parameters
		self.pi = torch.nn.Parameter(torch.ones(n_clusters)/n_clusters, requires_grad=True)
		self.mu_c = torch.nn.Parameter(torch.randn(n_clusters, out_channels), requires_grad=True)
		self.log_sigma2_c = torch.nn.Parameter(torch.randn(n_clusters, out_channels), requires_grad=True)

	def forward(self, x, edge_index):
		x = F.relu(self.gcn_shared(x, edge_index))
		mu = self.gcn_mu(x, edge_index)
		logvar = self.gcn_logvar(x, edge_index)
		return mu, logvar

	def predict_soft(self, z):
		det = 1e-2
		# import ipdb; ipdb.set_trace()
		yita_c = torch.exp(torch.log(self.pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, self.mu_c, self.log_sigma2_c)) + det
		return yita_c/yita_c.sum(dim=1).unsqueeze(1)

	def gaussian_pdfs_log(self,x,mus,log_sigma2s):
		G=[]
		for c in range(self.n_clusters):
			G.append(self.gaussian_pdf_log(x, mus[c], log_sigma2s[c,:]).view(-1,1))
		return torch.cat(G,1)

	def gaussian_pdf_log(self,x,mu,log_sigma2):
		c = -0.5 * torch.sum(np.log(np.pi*2) + log_sigma2 + (x - mu)**2/torch.exp(log_sigma2),1)
		return c

	@staticmethod
	def decode(z):
		A_pred = torch.sigmoid(torch.matmul(z,z.t()))
		return A_pred

def gmm_pretrain(vgae_model, adj, features, adj_label, y, weight_tensor, norm, epochs, lr, dataset_name, optimizer="Adam"):
	from sklearn.mixture import GaussianMixture

	if os.path.exists('pretrained_gmm/' + dataset_name + '.pt'):
		vgae_model.load_state_dict(torch.load('pretrained_gmm/' + dataset_name + '.pt'))
	else:
		if optimizer == "Adam":
			opti = torch.optim.Adam(vgae_model.parameters(), lr=lr)
		elif optimizer == "SGD":
			opti = torch.optim.SGD(vgae_model.parameters(), lr=lr, momentum=0.9)
		elif optimizer == "RMSProp":
			opti = torch.optim.RMSprop(vgae_model.parameters(), lr=lr)
		print('Pretraining......')
		
		# initialisation encoder weights
		nmi_best = 0
		gmm = GaussianMixture(n_components = vgae_model.encoder.n_clusters, covariance_type = 'diag')
		acc_list = []
		for _ in (epoch_bar := tqdm(range(epochs))):
			opti.zero_grad()
			z = vgae_model.encode(features, adj)
			x_ = vgae_model.encoder.decode(z)
			# import ipdb; ipdb.set_trace()
			loss = norm*F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
			loss.backward()
			opti.step()
			y_pred = gmm.fit_predict(z.detach().cpu().numpy())
			vgae_model.encoder.pi.data = torch.from_numpy(gmm.weights_)
			vgae_model.encoder.mu_c.data = torch.from_numpy(gmm.means_)
			vgae_model.encoder.log_sigma2_c.data =  torch.log(torch.from_numpy(gmm.covariances_))
			acc = utils.clustering_accuracy(y.cpu(), y_pred)
			nmi = sklearn.metrics.normalized_mutual_info_score(y.cpu(), y_pred)
			# acc_list.append(acc)
			epoch_bar.set_description_str('Loss = {:.4f}, Acc = {:.4f}, NMI = {:.4f}'.format(loss, acc, nmi))
			if (nmi > nmi_best):
			  nmi_best = nmi
			  # torch.save(vgae_model.state_dict(), 'pretrained_gmm/temp/' + f'nmi{nmi_best:.3f}' + dataset_name + f'_{datetime.now().strftime("%d%m_%H:%M")}.pt')
			  torch.save(vgae_model.state_dict(), 'pretrained_gmm/' + dataset_name + f'.pt')
		print("Best NMI : ",nmi_best)
		# return acc_list
