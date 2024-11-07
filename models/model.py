
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np
from models.GAT_layer import *
from models.multiGra import *
from utils.tools import load_cities
class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        even=self.even(x)
        odd=self.odd(x)
        even_len = even.size(1)
        odd_len = odd.size(1)
        flag=0
        if odd_len < even_len:
            pad_size = even_len - odd_len
            odd = F.pad(odd, (0, 0, 0, pad_size), mode='replicate')
            flag=1
        elif even_len < odd_len:
            pad_size = odd_len - even_len
            even = F.pad(even, (0, 0, 0, pad_size), mode='replicate')
            flag=2
        '''Returns the odd and even part'''
        return (even, odd),flag


class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True,
                 kernel = 5, dropout=0.5, groups = 1, hidden_size = 1, INN = True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1 #by default: stride==1 
            pad_r = self.dilation * (self.kernel_size) // 2 + 1 #by default: stride==1 

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),

            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd),flag = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)
            if flag==1:
                x_odd_update=x_odd_update[:,:,:-1]
            elif flag==2:
                x_even_update=x_even_update[:,:,:-1]
            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups , hidden_size, INN):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes = in_planes, splitting=True,
                 kernel = kernel, dropout=dropout, groups = groups, hidden_size = hidden_size, INN = INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)

class LevelSCINet(nn.Module):
    def __init__(self,in_planes, kernel_size, dropout, groups, hidden_size, INN):
        super(LevelSCINet, self).__init__()
        self.interact = InteractorLevel(in_planes= in_planes, kernel = kernel_size, dropout = dropout, groups =groups , hidden_size = hidden_size, INN = INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1) #even: B, T, D odd: B, T, D

class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN,layer_outputs=None):
        super().__init__()
        self.current_level = current_level


        self.workingblock = LevelSCINet(
            in_planes = in_planes,
            kernel_size = kernel_size,
            dropout = dropout,
            groups= groups,
            hidden_size = hidden_size,
            INN = INN)

        if layer_outputs is None:
            self.layer_outputs = [[] for _ in range(current_level+1)]
        else:
            self.layer_outputs = layer_outputs
        if current_level!=0:
            self.SCINet_Tree_odd=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN, self.layer_outputs)
            self.SCINet_Tree_even=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN, self.layer_outputs)
    
    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2) #L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len: 
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_,0).permute(1,0,2) #B, L, D
        
    def forward(self, x):
        x_even_update, x_odd_update= self.workingblock(x)
        self.layer_outputs[self.current_level].append(self.zip_up_the_pants(x_even_update, x_odd_update))

        if self.current_level ==0:
            a=self.zip_up_the_pants(x_even_update, x_odd_update)
            return a
        else:
            a=self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))
            return a
    
    def sort_and_get_layer_outputs(self):
        sorted_outputs = []
        for layer_output in self.layer_outputs:
            sorted_outputs.append(self.recursive_zip(layer_output))
        return sorted_outputs
    
    def recursive_zip(self, layer_output):
        if len(layer_output) == 2:
            return self.zip_up_the_pants(layer_output[0], layer_output[1])
        elif len(layer_output) == 1:  
            return layer_output[0]

        mid = len(layer_output) // 2
        first_half = layer_output[:mid]
        second_half = layer_output[mid:]
        
        return self.zip_up_the_pants(self.recursive_zip(first_half), self.recursive_zip(second_half))
    
    def clear_outputs(self): 
        self.layer_outputs[self.current_level] = []
        
        # clear for child nodes
        if self.current_level != 0:
            self.SCINet_Tree_odd.clear_outputs()
            self.SCINet_Tree_even.clear_outputs()

class EncoderTree(nn.Module):
    def __init__(self, in_planes,  num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.levels=num_levels
        self.SCINet_Tree = SCINet_Tree(
            in_planes = in_planes,
            current_level = num_levels-1,
            kernel_size = kernel_size,
            dropout =dropout ,
            groups = groups,
            hidden_size = hidden_size,
            INN = INN)
        
    def forward(self, x):
        self.SCINet_Tree.clear_outputs()
        x= self.SCINet_Tree(x)
        list_x=self.SCINet_Tree.sort_and_get_layer_outputs()
        return list_x

class MyModel(nn.Module):
    def __init__(self, output_len, input_len, input_dim = 9, hid_size = 1,
                num_levels = 3, concat_len = 0, groups = 1, kernel = 5, dropout = 0.5,
                 single_step_output_One = 0, input_len_seg = 0, positionalE = False, modified = True, RIN=False,
                 n_hid=48,alpha=0.2,n_class=18,n_heads=8,topk=8,device='cuda'):
        super(MyModel, self).__init__()
        self.device=device
        self.input_dim = input_dim
        self.out_feature=1
        self.ncity=int(input_dim/15)
        self.idx=torch.arange(self.ncity).to(self.device)
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN=RIN
        self.device=device
        self.static_adj=load_cities().to(self.device)

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified)
        
        self.global_embedding=global_graph_creator(int(input_dim/15),4,k=topk)
        self.scale_embedding=nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.out_att=nn.ModuleList()
        for i in range(self.num_levels+1):
            self.scale_embedding.append(local_graph_creator(int(input_dim/15),4,k=topk))
            self.attentions.append(nn.ModuleList([GraphAttentionLayer((input_dim*input_len)/self.ncity, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]))
            self.out_att.append(GraphAttentionLayer(n_hid * n_heads, self.ncity, dropout=dropout, alpha=alpha, concat=False))

        self.adj_reshape = nn.Linear(self.ncity*self.ncity,self.ncity*15)
        self.merge=nn.Conv2d(self.ncity*15,self.ncity*15,kernel_size=(num_levels+1,1))

        


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.filter1=nn.Linear(self.input_dim,4*self.input_dim)
        self.relu=nn.ReLU()
        self.dp=nn.Dropout(0.05)
        self.filter2=nn.Linear(4*self.input_dim,self.input_dim)
        self.norm=nn.LayerNorm(self.input_dim)
        self.fc1=nn.Linear(self.input_dim,self.out_feature)
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)

        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal

    def forward(self, x):
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        ### activated when RIN flag is set ###
        if self.RIN:
            print('/// RIN ACTIVATED ///\r',end='')
            means = x.mean(1, keepdim=True).detach()
            #mean
            x = x - means
            #var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x = x * self.affine_weight + self.affine_bias
        batch_size=x.shape[0]

        res1 = x
        x = self.blocks1(x)
        x.insert(0,res1)
        temp=x.copy()
        
        # GAT

        gmap,gemb=self.global_embedding(self.idx)
        gat_adj=[]
        adj=[0]*(self.num_levels+1)
        for i in range(self.num_levels+1):
            temp[i]=temp[i].reshape(batch_size,self.input_len,self.ncity,-1).permute(0,2,1,3)
            temp[i]=temp[i].reshape(batch_size,self.ncity,-1)

            adj[i]=self.scale_embedding[i](self.idx,gemb)

            temp[i]=F.dropout(temp[i],self.dropout,training=self.training)
            temp[i]=torch.cat([att(temp[i],adj[i]) for att in self.attentions[i]],dim=2) #(32,18,-1)
            temp[i] = F.dropout(temp[i], self.dropout, training=self.training)
            temp[i] = F.elu(self.out_att[i](temp[i],adj[i]))
            gat_adj.append(F.log_softmax(temp[i],dim=2))
        
        for i in range(self.num_levels+1):
            adj=gat_adj[i].view(batch_size,1,-1)
            adj=self.adj_reshape(adj)
            x[i]=x[i]*adj

        mg = torch.tensor([],device=self.device)
        for i in range(self.num_levels+1):
            mg = torch.cat((mg,x[i].unsqueeze(1)), dim =1)
        mg=self.merge(mg.permute(0,3,1,2))
        mg=mg.squeeze(-2).permute(0,2,1)
        mg=self.filter1(mg)
        mg=self.relu(mg)
        mg=self.dp(mg)
        mg=self.filter2(mg)
        x=self.norm(mg+res1)
        x=self.fc1(x)
        x =self.projection1(x)
                          
        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        return x,gmap