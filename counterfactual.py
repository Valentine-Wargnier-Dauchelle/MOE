import torch
import torch.nn as nn


class ActOptiLoss(nn.Module):
    """"""

    def __init__(self,model,img0):
        ''
        super(ActOptiLoss,self).__init__()
    
        self.l1    = nn.L1Loss (reduction='mean')
        self.model = model
        self.img0  = img0

    def forward(self, img):

        ri = self.l1(img,self.img0) 
        ri = ri.mean()

        o = self.model(img)
        o = o.mean()
        
        self.dataterm = ri.data.item()
        self.actterm  =  o.data.item()
        return 0.0001*ri + o



def activation_opti(model, img0, lr=1e-5, niter=1000, tresh=-15, mono=True):


    img = img0.clone()
    
    alpha = torch.nn.parameter.Parameter(torch.normal(0, 1e-3, img.size()).to(Device.device))
    alpha.requires_grad = True
    optimizer = torch.optim.Adam([alpha], lr)

    aoloss = ActOptiLoss(model, img0)
    
    if mono:
        act = torch.nn.Softplus()
    else:
        act= torch.nn.Identity()

    stop = False
    i = 0 
    while i < niter and (not stop or i < 10):
        optimizer.zero_grad()

	aol = aoloss(img0-act(alpha))
        aol.backward(retain_graph=True)
        optimizer.step()

        stop = (aoloss.actterm<thresh)
        i += 1 
    
    return act(alpha)







