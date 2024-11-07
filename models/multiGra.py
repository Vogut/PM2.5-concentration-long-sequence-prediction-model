
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np
# from GAT_layer import *
from models.GAT_layer import *


class local_graph_creator(nn.Module):
    def __init__(self,num_nodes,dim,k=10,alpha=3):
        super(local_graph_creator,self).__init__()
        self.emb=nn.Embedding(num_nodes,dim)
        self.alpha=alpha
        self.k=k
        self.fc1=nn.Linear(dim,dim)
        # self.fc2=nn.Linear(dim,dim)

    def forward(self,idx,gEmb):
        vec1=self.emb(idx)
        vec1=torch.tanh(self.alpha*self.fc1(vec1))
        # vec2=self.alpha*self.fc2(gEmb)
        a = torch.mm(vec1,gEmb.transpose(1,0))-torch.mm(gEmb,vec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))

        mask = torch.zeros(idx.size(0), idx.size(0),dtype=torch.float64).to(idx.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        # print(mask)
        return adj*mask   

class global_graph_creator(nn.Module):
    def __init__(self,num_nodes,dim,k=10,alpha=3):
        super(global_graph_creator,self).__init__()
        self.emb=nn.Embedding(num_nodes,dim)
        self.alpha=alpha
        self.k=k
        self.fc1=nn.Linear(dim,dim)
        self.fc2=nn.Linear(dim,dim)

    def forward(self,idx):

        vec1=self.emb(idx)
        vec2=self.emb(idx)
        vec1=torch.tanh(self.alpha*self.fc1(vec1))
        vec2=torch.tanh(self.alpha*self.fc2(vec2))
        a_temp= (torch.mm(vec1,vec2.transpose(1,0))+torch.mm(vec2,vec1.transpose(1,0)))/2
        a = a_temp -torch.diag_embed(torch.diag(a_temp))
        adj = F.relu(torch.tanh(self.alpha*a))

        mask = torch.zeros(idx.size(0), idx.size(0),dtype=torch.float64).to(idx.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))

        out_adj=adj*mask
        return out_adj,vec1  

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.adj_reshape = nn.Linear( 18 * 18 , 18*15)     

    def forward(self, input_x, adj):
        # # -----------------------------------
        """ GAT Part """
        x = F.dropout(input_x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        gat_adj = F.log_softmax(x, dim=1)
        # # ------------------------------------

        # # -----------
        ''' visualize adjacent matrix (unsupervised) '''
        # save_tensor_to_img(gat_adj.data.cpu().numpy(), 'gat_adj')
        # # -----------
        
        # # ------------------------------------
        ''' GAT cooperate with sequence '''
        tmp_adj = gat_adj.view(1, -1)
        tmp_adj = self.adj_reshape(tmp_adj)
        input_x = input_x.reshape(1, -1, 18*15)
        out_x = input_x * tmp_adj
        
        return  out_x,gat_adj