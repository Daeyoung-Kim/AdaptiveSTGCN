import torch
import torch.nn as nn

from model import layers

class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND GND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout
    
    # G: Graph Convolution Layer (ChebGraphConv)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConv, self).__init__()

        self.st_block1 = layers.STConvBlock(args['Kt'], args['Ks'], n_vertex, blocks[0][-1], blocks[0+1], args['act_func'], args['graph_conv_type'], args['enable_bias'], args['droprate'])
        self.st_block2 = layers.STConvBlock(args['Kt'], args['Ks'], n_vertex, blocks[1][-1], blocks[1+1], args['act_func'], args['graph_conv_type'], args['enable_bias'], args['droprate'])
        Ko = args['n_his'] - (2) * 2 * (args['Kt'] - 1)
        self.Ko = Ko
        
        self.adap_block1 = layers.AdapBlock(args['Ks'],n_vertex, blocks[1+1][-1], blocks[1+2][0], args['graph_conv_type'],args['enable_bias'], args['droprate'])
        self.adap_block2 = layers.AdapBlock(args['Ks'],n_vertex, blocks[2+1][-1], blocks[2+2][0], args['graph_conv_type'],args['enable_bias'], args['droprate'])
        
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args['act_func'], args['enable_bias'], args['droprate'])
        elif self.Ko == 0:
            
            raise NotImplementedError(f'ERROR: Ko is 0.')
            
            # self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            # self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            # self.relu = nn.ReLU()
            # self.leaky_relu = nn.LeakyReLU()
            # self.silu = nn.SiLU()
            # self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x, gso, adp_gso):
        x = self.st_block1(x, gso)
        x = self.st_block2(x, gso)
        x = self.adap_block1(x, adp_gso)
        x = self.adap_block2(x, adp_gso)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            
            raise NotImplementedError(f'ERROR: Ko is 0.')
            
            # x = self.fc1(x.permute(0, 2, 3, 1))
            # x = self.relu(x)
            # x = self.fc2(x).permute(0, 3, 1, 2)
        
        return x

'''
class STGCNGraphConv(nn.Module):
    # STGCNGraphConv contains 'TGTND TGTND TNFF' structure
    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.
    # Be careful about over-smoothing.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        return x
'''