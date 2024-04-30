"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.
Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)  每个节点在meta-path维度的均值
        beta = torch.softmax(w, dim=0)                 # (M, 1)  归一化   
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1) 拓展到N个节点上的meta-path的值
        # (beta * z)=>所有节点在meta-path上的attention值，
        # (beta * z).sum(1)=>节点最终的值
        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)): #meta-path Layers：两个meta-path的维度是一致的
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)  #语义级attention
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1)) #每个meta-path对应一个GAT
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)每个节点对应到meta-path下的每个节点的embedding值
        #聚合meta-path下，每个节点最终的输出值
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size,out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
            #输出的是：节点、meta-path的值，embedding;
            #return:节点HAN后输出的embedding
        return h
        # return self.predict(h)  #HAN输出节点embedding后接一个Linear层