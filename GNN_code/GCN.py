# Reference: https://zhuanlan.zhihu.com/p/422380707

import dgl
import torch
from dgl.data import CoraGraphDataset

def build_graph_test(self):
    """a demo graph: just for graph test
    """
    src_nodes = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6])
    dst_nodes = torch.tensor([1, 2, 0, 2, 0, 1, 3, 4, 5, 6, 2, 3, 3, 3])
    graph = dgl.graph((src_nodes, dst_nodes))
    # edges weights if edges has else 1
    graph.edata["w"] = torch.ones(graph.num_edges())
    return graph

def build_graph_cora(self):
    # Default: ~/.dgl/
    data = CoraGraphDataset()
    graph = data[0]

    return graph

###########################################

from utils import preprocess_adj

def get_adj(self, graph):
    graph = self.add_self_loop(graph)
    # edges weights if edges has weights else 1
    graph.edata["w"] = torch.ones(graph.num_edges())
    adj = coo_matrix((graph.edata["w"], (graph.edges()[0], graph.edges()[1])),
                     shape=(graph.num_nodes(), graph.num_nodes()))

    #  add symmetric edges
    adj = self.convert_symmetric(adj, sparse=True)
    # adj normalize and transform matrix to torch tensor type
    adj = preprocess_adj(adj, is_sparse=True)

    return adj

############################################

import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #  D^(-1/2)AD^(-1/2)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

############################################

class GraphConvolution(Module):
    def __init__(self, in_features_dim, out_features_dim, activation=None, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.activation = activation
        self.weight = Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, infeatn, adj):
        '''
        infeatn: init feature(H，上一层的feature)
        adj: A
        '''
        support = torch.spmm(infeatn, self.weight)  # H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
        output = torch.spmm(adj, support)  # A*H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output

##########################################

class GCN(Module):
    def __init__(self, nfeat, nhid, nclass, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConvolution(nfeat, nhid, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConvolution(nhid, nhid, activation=activation))
        # output layer
        self.layers.append(GraphConvolution(nhid, nclass))
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, adj):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, adj)
        return h
