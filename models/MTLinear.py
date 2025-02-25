import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from data_provider.data_factory import data_provider
import copy
from layers.RevIN import RevIN
from layers.Linear_layers import DLinear, Linear, NLinear, RLinear
from sklearn.cluster import AgglomerativeClustering




class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs, init_weights=None, seq_len = None):
        super(Model, self).__init__()

        self.configs = configs
        self.seq_len  = seq_len

        if seq_len is None:
            self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in


        self.cluster_dist= configs.cluster_dist

        self.variate_clusters  = self.set_variate_clusters()
        print("self.args.variate_clusters")
        print(self.variate_clusters)

        # for DLinear layer
        self.kernel_size = 25

        self.layer_type = configs.layer_type

        self.model_name = "MTLinear"
        self.layer_dict = {'DLinear':DLinear, 'Linear':Linear, 'NLinear':NLinear, 'RLinear': RLinear}
        self.layers = {}
        self.layers_inds = {}

        self.pred_i_last = False
        if init_weights is not None:
            self.pred_i_last = True
            self.seq_len = init_weights.weight.shape[1]
            self.pred_len = init_weights.weight.shape[0]

        self.create_h_layers(init_weights)
        print("layers")
        print(self.layers)
        self.layers = nn.ModuleDict(self.layers)

        # print layer names
        print("layer names")
        for name, param in self.named_parameters():
            print(name)
        print()

    def set_variate_clusters(self, x = None, flag = "train"):
        if x is None:
            dataset, _ = data_provider(self.configs, flag)
            x = dataset.data_x.T

        # x shape (n_variates,seq_len)
        inds = np.arange(0,self.configs.enc_in)

        mapping = {i:int(ind) for i,ind in enumerate(inds)}
        corr = np.corrcoef(x)

        if self.cluster_dist%1!=0 or self.cluster_dist==0:
            distance_threshold = self.cluster_dist
            n_clusters = None
        else:
            distance_threshold = None
            n_clusters = int(self.cluster_dist)

        clustering = AgglomerativeClustering(compute_full_tree =True,
                                                                    n_clusters=n_clusters,
                                                                    distance_threshold = distance_threshold,
                                                                    affinity = 'precomputed',
                                                                    #metric = 'precomputed',
                                                                    linkage='complete',
                                                                    compute_distances =True).fit(-corr+1)
                                                    
        clustering_labels = clustering.labels_
        self.configs.k_tasks = clustering.n_clusters_
        cl_labels = [np.where(clustering_labels == k)[0].tolist() for k in range(0,self.configs.k_tasks)]
        cl_labels = [[mapping[ind] for ind in cl] for cl in cl_labels]
        return cl_labels
        
    
    def create_h_layers(self, init_weights=None ):
        for j in range(len(self.variate_clusters)): 
            self.layers[str(j)] = self.layer_dict[self.layer_type](self.seq_len,self.pred_len) 
            if init_weights is not None:
                self.layers[str(j)] = copy.deepcopy(init_weights)
            self.layers_inds[str(j)] = self.variate_clusters[j]

    def forward(self, x ):
        if self.pred_i_last:
            x_out = torch.zeros([x.size(0),self.channels,self.pred_len],dtype=x.dtype).to(x.device)
        else:
            x_out = torch.zeros([x.size(0),self.pred_len,self.channels],dtype=x.dtype).to(x.device)
        for j in range(len(self.variate_clusters)):
            # route the data to the correct layer
            task_inds = self.variate_clusters[j]
            if self.pred_i_last:
                x_out[:,task_inds,:] = self.layers[str(j)](x[:,task_inds,:] )
            else:
                x_out[:,:,task_inds] = self.layers[str(j)](x[:,:,task_inds] ) 
        return x_out
