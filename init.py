import torch.nn
import torch.nn as nn

import numpy as np

import utils

class WeightRescaler(nn.Module):

    def __init__(self, layer, bias=False, eps=1e-8, scalar=False, print_mode=None):
        super().__init__()
        self.orig       = layer
        self.print_mode = print_mode
        self.mucov      = MuCovImg(centered=bias, eps=0, diagonly=True)
        self.cov_fn     = MuCovImg(centered=True)
        self.eps        = eps
        self.bias       = bias
        if scalar:
            self.dim = (0,1)
        else:
            self.dim = 0


    def forward(self,x):
        print(f'---------------------------------- Weight rescaler B: {x.shape} ')
        print(self.orig)
        printsuf = self.print_mode.suf if self.print_mode is not None else ""

        if self.print_mode.only and self.print_mode.isinf:
            y = self.orig(torch.zeros_like(x))

            print(f"CORRnoactiv{printsuf}___{self.orig.__class__.__name__}___inf")
            print(f'rescalebefore{printsuf} {self.orig.__class__.__name__} {x.shape} {y.shape} inf inf')
            return y

        y = self.orig(x)

        # compute and print correlation between features
        if self.print_mode.only and y.shape[1]>1 and len(y.shape)>3:

            cov = self.cov_fn.mucov0(y[0].unsqueeze(0))[1][0]
            dcov           = torch.diag( 1.0 / (torch.sqrt(torch.diag(cov))+1.0e-10) )

            corrMat        = torch.mm(dcov,torch.mm(cov,dcov)) # docv * cov * dcov
            corrb          = torch.sum(corrMat.triu(diagonal=1)) / ( (cov.shape[0]*(cov.shape[0]-1))/2 ) 

            print(f"CORRnoactiv{printsuf}___{self.orig.__class__.__name__}___{corrb}")



        # mean/var over space
        mu,   cov = self.mucov.mucov(y)

        # mean over batch
        cov = torch.mean(cov,dim=self.dim,keepdims=False)
        mu  = torch.mean(mu,dim=self.dim,keepdims=False).squeeze()
        
        print(f'rescalebefore{printsuf} {self.orig.__class__.__name__} {x.shape} {y.shape} {torch.mean(mu)} {torch.sqrt(torch.mean(cov))}')

        if self.print_mode is not None and self.print_mode.only == 'yes':
            self.print_mode.isinf = torch.any(torch.isinf(y)) or torch.any(torch.isnan(y))
            return y

        if self.bias and hasattr(self.orig, "bias") and self.orig.bias is not None:
            print(f'################################################################  {mu.shape} {cov.shape} {self.orig.bias}')
            newbias   =             - mu / torch.sqrt(cov+self.eps)

            if len(newbias.shape)==0 and len(self.orig.bias.data.shape)>0: # newbias is scalar
                newbias = torch.ones_like(self.orig.bias.data) * newbias

        while(len(cov.shape)<len(self.orig.weight.shape)):
            cov = cov.unsqueeze(-1)
            mu  = mu .unsqueeze(-1)
        
        if hasattr(self.orig, "pos_fn_inv"): ##right inverse for specific layer
            print(self.orig.fn)
            newweight = self.orig.pos_fn_inv(self.orig.weight, cov)
        else:
            newweight = self.orig.weight / torch.sqrt(cov+self.eps)

        def setval(attname, val):
            if ( hasattr(self.orig,'parametrizations') and
                hasattr(self.orig.parametrizations,attname) ):
                setattr(self.orig, attname, val)
            else:
                setattr(self.orig, attname, nn.Parameter( val ) )

        setval('weight',newweight)
        if self.bias and hasattr(self.orig, "bias") and self.orig.bias is not None:
            setval('bias',newbias)

        cov = cov.unsqueeze(0).squeeze(-1)
        mu  = mu .unsqueeze(0).squeeze(-1)
        ynew = ( y - mu )
        ynew = ( y - mu ) / torch.sqrt(cov+self.eps)

        if self.print_mode.check:
            z = self.orig(x)
            mu,   cov = self.mucov.mucov(z)
            cov = torch.mean(cov,dim=0,keepdims=False)
            print('muB')
            print(mu)
            print('covB')
            print(torch.mean(cov))
            print(cov)
            print(f'rescaleafter{printsuf} {self.orig.__class__.__name__} {x.shape} {y.shape} {torch.mean(mu)} {torch.sqrt(torch.mean(cov))} {torch.max(torch.abs(z-ynew))}')
        print('---------------------------------- Weight rescaler E')
        return ynew

#############################################################################

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

###################################
# replace conv / linear layer with WeightRescaler(conv) layer
def addWeightRescaler(model,bias=False, scalar=False, print_mode=None):
    # loop over layers
    for name, module in model.named_modules():
       
        # linear / conv layer : param with Softplus
        if (    isinstance(module, nn.Linear) 
             or isinstance(module, nn.Conv1d) 
             or isinstance(module, nn.Conv2d) 
             or isinstance(module, nn.Conv3d) 
             or isinstance(module, nn.parameter.Parameter)
            ) :
            
            layer_orig = get_layer(model, name)
            layer_new  = WeightRescaler(layer_orig,bias, scalar=scalar, print_mode=print_mode)
            set_layer(model, name, layer_new)

###################################
# replace WeightRescaler layers by orig layers
def removeWeightRescaler(model):
    
    # loop over layers
    for name, module in model.named_modules():
                # linear / conv layer : param with Softplus
        if (    isinstance(module, WeightRescaler) ) :
            wr_layer = get_layer(model, name)
            set_layer(model, name, wr_layer.orig)


###################################
def initByWeightRescaling(model,ishape,bias=False, scalar=False, verbose=0):

    with torch.no_grad():
        
        print_mode = utils.Generic(only=True, suf='orig', isinf=False, check=True, corr=True)
        print_mode.check = False
    
        imgtrain = torch.randn(ishape, dtype=torch.float32, device=next(model.parameters()).device)
        imgtest  = torch.randn(ishape, dtype=torch.float32, device=next(model.parameters()).device)
        
        print('------------------ ADD WEIGHT RESCALER')
        addWeightRescaler(model, bias=bias, scalar=scalar, print_mode=print_mode)

        # only print orig
        if verbose>0:
            print_mode.only  = True
            print_mode.suf   = '_orig'
            print_mode.isinf = False
            print('------------------ WEIGHT RESCALER: test pre rescale')
            out = model(imgtest)

        # rescale on train
        print_mode.only = False
        print_mode.suf  = '_resc'
        print_mode.isinf = False
        print('------------------ WEIGHT RESCALER: rescale')
        out = model(imgtrain)
        
        # only print after rescale
        if verbose>0:
            print_mode.only  = True
            print_mode.suf   = '_after'
            print_mode.isinf = False
            print('------------------ WEIGHT RESCALER : test post recale')
            out = model(imgtest)

        print('------------------ REMOVE WEIGHT RESCALER')
        removeWeightRescaler(model)
        return not(torch.any(out.isnan()))

##############################################################################################
##############################################################################################

class MuCovImg(nn.Module):
    def __init__(self, centered=False, eps=0, diagonly=False):
        super().__init__()
        self.centered = centered
        self.eps      = eps
        self.diagonly = diagonly

    def mucov0(self,img):
        ''' compute spatial mean and correlation '''
        nc   = img.shape[1]
        dims = tuple(range(1,len(img.shape)-1))
        c    = torch.zeros((img.shape[0],nc,nc), dtype=img.dtype, device=img.device)

        if self.centered:
            mudims = tuple(range(2,len(img.shape)))
            mu = torch.mean(img,dim=mudims,keepdim=True)
            for i in range(nc):
                for j in range(i,nc):
                    c[:,i,j] = torch.mean( (img[:,i,...]-mu[:,i,...]) * (img[:,j,...]-mu[:,j,...]), dim=dims)
                    c[:,j,i] = c[:,i,j]
        else:
            mu = torch.zeros((img.shape[0],nc), dtype=img.dtype, device=img.device)
            for i in range(nc):
                for j in range(i,nc):
                    c[:,i,j] = torch.mean(  img[:,i,...]              *  img[:,j,...]             , dim=dims)
                    c[:,j,i] = c[:,i,j]
        return mu, c

    def mucov(self,img):
        
        N = np.prod(list(img.shape[2:]))

        ''' compute spatial mean and correlation '''
        if self.centered:
            mudims = tuple(range(2,len(img.shape)))
            mu = torch.mean(img,dim=mudims,keepdim=True)
            if self.diagonly:
                c  = torch.einsum('bc...,bc...->bc', [img-mu,img-mu]) / N
            else:
                c  = torch.einsum('bc...,bC...->bcC',[img-mu,img-mu]) / N
        else:
            mu = torch.zeros(img.shape[0:2], dtype=img.dtype, device=img.device)
            if self.diagonly:
                c  = torch.einsum('bc...,bc...->bc', [img,img]) / N
            else:
                c  = torch.einsum('bc...,bC...->bcC',[img,img]) / N
        return mu, c


