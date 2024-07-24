
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.parametrize as torchparam

import utils

import architecture

############################################


class MoE(nn.Module):
    """ 
    composition of 2 network, the second is positive
    """

    def __init__(self, dim, enc, disc, argmoe, nullbiasE=True, nullbiasP=True):
        super().__init__()
        self.dim =dim
        self.argmoe = argmoe
        self.enc   = enc

        self.pos    = positiveWeightParam(disc, argmoe.beta, nullbias=nullbiasP) if not argmoe.notpos else disc

        print(' -------------- argmoe')
        print(argmoe)


        if nullbiasE:
            for name, module in self.enc.named_modules():
                if (    isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) 
                     or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) ) :
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)
                        module.bias.requires_grad = False

                elif (  isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.GroupNorm) 
                     or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm) 
                     or isinstance(module, nn.InstanceNorm3d) or isinstance(module, nn.BatchNorm3d) 
                     ):  
                    torch.nn.init.constant_(module.weight, 1)
                    module.weight.requires_grad = False
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias,   0)
                        module.bias.  requires_grad = False
        
        if argmoe.moeKLfeat.nbin>0:
            self.histKLloss = DiffHistKL(argmoe.moeKLfeat.nbin,argmoe.moeKLfeat.hloss)
            

    @staticmethod
    def add_argument_to(parser):

        def moeparse(cmdline):
            a = utils.Generic()
            s = cmdline.split('/')
            a.notpos    = (s[0]=="none")                    # do not convert discrim to positive but add pre-positive network (for comparison)
            a.beta       = 1 if a.notpos else float(s[0]) 
            a.etype      =       s[1]
            a.nfeatmid   = int(  s[2])

            a.amweight   = float(s[3])


            if a.etype == 'CN':
                a.ltype, a.ln, a.lks, a.ldil, a.lno = architecture.C1Net._parse(s[4:])
                a.nfeatmid   = a.ln[-1]

            return a

        parser.add_argument('--moe',                      
                            default="0/I/0/0",  
                            type=moeparse, 
                            help= 'train moe decomposition'
                            + ' beta/Etype/nfeatmid/negweight[/Cnet]  : '
                            + 'beta<0unused, >0:positive parametrization (generally 1) / Etype in I, C, CN : Identity, Conv, Conv Network/ Nb filters at the end of E)'
                            + ' Cnet: if Etype==CN : C1Net definition'
                           )
        
        def moeparsekl(cmdline):
            a = utils.Generic()
            s = cmdline.split('/')
            a.nbin   = int(  s[0])
            a.weight = float(s[1])
            a.hloss  = s[2]
            return a
        parser.add_argument('--moeKLfeat',     default="0/0/KL", type=moeparsekl,   help='Number of bins/Loss weight/KL or hellinger distance')
        


    def negfeatloss(self, efeat):
        return torch.mean(torch.nn.functional.relu(efeat)) 
 
    def gradientloss(self, efeat, output):
        output = torch.mean(output, list(range(1, len(output.size())))) 
        attributions = torch.autograd.grad(output, efeat, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        loss = torch.sum(attributions)
        return loss, attributions
        
    def klfeatloss(self, efeat0, efeat1):
        return self.histKLloss(efeat0, efeat1)

    
    def forward(self,x):
        self.efeat = self.enc(x)
        
        res        = self.pos(self.efeat)
        self.feat  = self.pos.feat

        return res 




#############################################################################################
#############################################################################################

class PositiveParam(nn.Module):
    def __init__(self,beta,fact=1):
        super().__init__()
        self.e    = beta
        self.e2   = beta*beta
        self.fact = fact
    def forward(self,w):
        w2 = w*w
        return self.fact * w2 / ( torch.sqrt(w2+self.e2) + self.e )
    def right_inverse(self, w):
        a = torch.abs(w)
        return torch.sqrt(a*a+2*a*self.e) # => foraword o rightinverse = abs(w)

######################################################################################
# to replace LeakyRelu with a convex activation for first channels, concave for last channels
class SplitLeakyRelu(nn.Module):
    def __init__(self, negative_slope=0.01,inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace        = inplace
    def forward(self,x):
        even =  torch.nn.functional.leaky_relu( x[:,::2,...],   negative_slope=self.negative_slope, inplace=self.inplace)
        odd  = -torch.nn.functional.leaky_relu(-x[:,1::2,...], negative_slope=self.negative_slope, inplace=self.inplace)
        t = torch.zeros_like(x)
        t[:,::2,...]  = even
        t[:,1::2,...] = odd
        return t


#######################################################################################################
###################################
# convert a network to a positive network:
# linear/conv : param with softplus
# normalization layer: replace with identity 
# activation replaced with SplitLeakyRelu 
def positiveWeightParam(model,beta=1, nullbias=True):

    ###################################
    def get_layer(model, name):
        layer = model
        for attr in name.split("."):
            layer = getattr(layer, attr)
        return layer
    #################################
    def set_layer(model, name, layer):
        try:
            attrs, name = name.rsplit(".", 1)
            model = get_layer(model, attrs)
        except ValueError:
            pass
        setattr(model, name, layer)
    #################################

    normlosses = []
    
    # loop over layers
    for name, module in model.named_modules():
        
        # linear / conv layer : param with Softplus
        if (    isinstance(module, nn.Linear) 
             or isinstance(module, nn.Conv1d) 
             or isinstance(module, nn.Conv2d) 
             or isinstance(module, nn.Conv3d) 
           ) :
            torchparam.register_parametrization(module, "weight", PositiveParam(beta=beta, fact=1)) 
            

            #if nullbias and hasattr(module,'bias'):
            if nullbias and module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
                module.bias.requires_grad = False

        # activation: replace with SplitLeakyRelu: monotonic but mix of convex and concave
        elif (     isinstance(module, nn.ELU)           or isinstance(module, nn.Hardshrink)    
                or isinstance(module, nn.Hardsigmoid)   or isinstance(module, nn.Hardtanh)      
                or isinstance(module, nn.Hardswish)     or isinstance(module, nn.LeakyReLU)   
                or isinstance(module, nn.LogSigmoid)    or isinstance(module, nn.PReLU)         
                or isinstance(module, nn.ReLU)          or isinstance(module, nn.ReLU6)         
                or isinstance(module, nn.RReLU)         or isinstance(module, nn.SELU)          
                or isinstance(module, nn.CELU)          or isinstance(module, nn.GELU)          
                or isinstance(module, nn.Sigmoid)       or isinstance(module, nn.SiLU)          
                or isinstance(module, nn.Mish)          or isinstance(module, nn.Softplus)      
                or isinstance(module, nn.Softshrink)    or isinstance(module, nn.Softsign)      
                or isinstance(module, nn.Tanh)          or isinstance(module, nn.Tanhshrink)    
                or isinstance(module, nn.Threshold)     or isinstance(module, nn.GLU)           
             ):  
            
            set_layer(model, name, SplitLeakyRelu())

        # normalization layer : remove them (replace with nn.Identity )
        # substracting the mean in norm layer => not monotonic
        elif (  isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.GroupNorm) 
             or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm) 
             or isinstance(module, nn.InstanceNorm3d) or isinstance(module, nn.BatchNorm3d) 
             ):  
           
            set_layer(model, name, nn.Identity())

    return model

#######################################################################################################
#####################################################################
class DiffHist(nn.Module):
    def __init__(self, hmin, hmax, nbin):
        super().__init__()
        self.nbin = nbin
        self.setminmax(hmin,hmax)

    def setminmax(self,hmin,hmax):
        self.hmin = hmin
        self.hmax = hmax
        self.dh   = (hmax-hmin)/(self.nbin-1)

    def forward(self, img):

        img  = img.flatten()
        
        # keep only pixel in min/max
        keep = torch.logical_and(img>=self.hmin, img<=self.hmax)
        img  = img[keep] - self.hmin
        if img.shape[0]==0:
            return None

        idx  = torch.floor( img / self.dh)
        dimg = img/self.dh - idx

        h = torch.zeros((self.nbin+1), device=img.device, dtype=img.dtype)
        h.index_add_(dim=0, index=idx.int(),   source=1-dimg)
        h.index_add_(dim=0, index=idx.int()+1, source=  dimg)
        return h[0:self.nbin]


#####################################################################
class DiffHistKL(nn.Module):
    def __init__(self, nbin, hloss='KL'):
        super().__init__()
        self.nbin = nbin
        self.diffhist = DiffHist(-1,0,nbin)

        self.kldiv = torch.nn.KLDivLoss(log_target=True)
        self.hloss = hloss

    def forward(self, img0, img1):

        min0 = torch.min(img0) 
        self.diffhist.setminmax(hmin=min0,hmax=0)

        h0 = self.diffhist(img0)

        if h0 is None: return img0.flatten()[0]*0
        
        h1 = self.diffhist(img1)
        if h1 is None: return img1.flatten()[0]*0
 
        eps = 1.0e-10
        h0 = (h0+eps) / (h0.sum()+eps)
        h1 = (h1+eps) / (h1.sum()+eps)
        if self.hloss=='KL':
            #kullback leibler
            return self.kldiv(torch.log((h1+eps)/h1),torch.log((h1+eps)/h0))
        else:
            # hellinger distance
            return torch.linalg.vector_norm( torch.sqrt(h1)-torch.sqrt(h0), ord=2) / self.nbin


