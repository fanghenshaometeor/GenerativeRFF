import torch
import torch.nn as nn
import numpy as np

class netG(nn.Module):
    """
    GRFF block.
    A generator for transform an arbitrary input noise distribution into
        some distribution of kernel.
    The weights sampled from the generated distribution are incorporated 
        together with data to build the corresponding generated random Fourier
        features.
    """
    def __init__(self, nx, D, nz = 100, nfm = 32):
        """
        ATTRIBUTES:
            nx: dimension of generated weights
            nz: dimension of input noise
            nfm: feature map factor
            D: number of input noises, also half of the dimension of generated
                random Fourier features
        """
        super(netG, self).__init__()
        self.D = D
        self.nx = nx
        self.generator = nn.Sequential(
            # ---------------------------------
            nn.Linear(nz, nfm * 2),
            nn.BatchNorm1d(nfm * 2),
            nn.LeakyReLU(0.01, inplace=True),
            # ---------------------------------
            nn.Linear(nfm * 2, nfm),
            nn.BatchNorm1d(nfm),
            nn.LeakyReLU(0.01, inplace=True),
            # ---------------------------------
            nn.Linear(nfm, nfm),
            nn.BatchNorm1d(nfm),
            nn.LeakyReLU(0.01, inplace=True),
            # ---------------------------------
            nn.Linear(nfm, nx),
            nn.Tanh()
        )
    def forward(self, noise, b_data):
        """
        ATTRIBUTES:
            noise: a group of arbitrary noise input
            b_data: a batch of data input
        OUTPUTS:
            z: generated random Fourier features
            w: generated sampled weights
        """
        D = self.D
        self.w = self.generator(noise)
        # ---------- building RFF without random bias
        z_1 = np.sqrt(1/D) * torch.cos(b_data.mm(self.w.t()))
        z_2 = np.sqrt(1/D) * torch.sin(b_data.mm(self.w.t()))
        self.z = torch.cat((z_1, z_2),1)
        return self.z, self.w

class GRFFNet(nn.Module):
    """
    An MLP included with the GRFF block.
    Based on the assumption that the weights in neural network follow some
        distribution, hence, we adopt a generator to generate weights following
        some distribution. The generated weights will be used as the traditional
        weights in an MLP.    
    """
    def __init__(self, GRFFBlock, num_classes, num_layers, d, D):
        """
        ATTRIBUTES:
            GRFFBlock: the GRFF generator
            num_classes: number of classes
            num_layers: number of layers with GRFF blocks in an MLP, when initialize, num_layers=1
            d: dimension of input data
            D: half of the dimension of generated random Fourier features
            
            GRFF: cascaded GRFF blocks. MLP with GRFF block in each layer
            fc: the extra final layer in the MLP
            
            when initialize, only one deep layer
        """
        super(GRFFNet, self).__init__()
        self.d = d
        self.D = D
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.GRFF = nn.Sequential(*[GRFFBlock(d, D[0], nz=100, nfm=64)])
        self.fc = nn.Linear(D[-1]*2, num_classes)
    
    def _add_layer(self, GRFFBlock, layer_index):
        """
        Add a GRFF block/layer
        
        """
        D = self.D
        num_layers = self.num_layers
        GRFFBlocks = self.GRFF
        
        layer = GRFFBlock(2*D[layer_index-1], D[layer_index], nz=100, nfm=D[layer_index-1])
        GRFFBlocks.add_module('%d'%(num_layers), layer)
        
        self.num_layers = num_layers+1
        self.GRFF = GRFFBlocks        
        return       
    
    def forward(self, x, noise):
        """
        ATTRIBUTES:
            x: input data
            noise: a group of input noises, each group corresponding to
                the input of one GRFF block
        OUTPUTS:
            out: the logits output of the final fc layer
            features: generated random Fourier features of each GRFF block
            weights: generated sampled weights of each GRFF block
        
        When forward, in the inverse direction
        For example,
            single layer well trained => GRFF[0] well trained
            add layer                 => GRFF[0] well trained, GRFF[1] random
            two layers forward        => GRFF[1] -> GRFF[0]
        """
        num_layers = self.num_layers
        
        for idx in range(num_layers):
            
            layer = self.GRFF[idx]
            x, w = layer.forward(noise[idx], x)
            
        out = self.fc(x)
        return out