import torch
import torch.nn as nn

import copy
import distutils.util

import moe
import utils

#############################################################################################
#############################################################################################

class ConvR(nn.Module):
    """
    """
    def __init__(self, dim, ni, no, ks, norm, activation, stride=1, **kwargs):
        super().__init__()
        
        self.ni = ni
        self.no = no
        self.dfeat=[]

        layers = []

        # conv
        if   dim==1: c = nn.Conv1d(ni, no, ks, stride=stride, **kwargs) 
        elif dim==2: c = nn.Conv2d(ni, no, ks, stride=stride, **kwargs) 
        elif dim==3: c = nn.Conv3d(ni, no, ks, stride=stride, **kwargs) 
        else: 
            raise ValueError("dim={}, should in [1,3]".format(dim))
        layers.append( c )

        # activation
        if   activation== "LeakyReLu": layers.append(nn.LeakyReLU(0.01,inplace=True))
        elif activation== "Sigmoid"  : layers.append(nn.Sigmoid  (                 ))
        elif activation== "Tanh"     : layers.append(nn.Tanh  (                 ))
        elif activation== "ReLu"     : layers.append(nn.ReLu     (     inplace=True))

        if norm=="inorm":
            if   dim==1: layers.append(nn.InstanceNorm1d(no, momentum=0, affine=True))
            elif dim==2: layers.append(nn.InstanceNorm2d(no, momentum=0, affine=True))
            elif dim==3: layers.append(nn.InstanceNorm3d(no, momentum=0, affine=True))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        self.dfeat = [ self.features(x) ]
        return self.dfeat[0]


##########################################################################################################

class CycleGAN_Discriminator(nn.Module):
    """
    Inspired from CycleGan discriminator (Zhu et al., 2017)      
    
    """
    
    def __init__(self, n_init_features, threeD=True, filter_config=(64, 128, 256, 512), drop_rate=0, init="kaiming", ortho_gain=0.8, pool="", act={"name":"LeakyReLu", "par":0.2}, poollast="amax", kernel=4, groups=1):
        super().__init__()
        
        if pool!="":
            stride=1
        else:
            stride=2
        
        if threeD:
            self.mode='trilinear'
            self.dim = [1]*5
        else:
            self.mode='bilinear'
            self.dim = [1]*4

        actname  = act.get('name','LeakyReLu')
        actparam = act.get('par',0.2)

        self.conv1 = Conv_layer(n_init_features, filter_config[0],stride,actname, False,threeD, pool, actparam=actparam, kernel=kernel, groups=groups)
        self.conv2 = Conv_layer(filter_config[0],filter_config[1],stride,actname, True, threeD, pool, actparam=actparam, kernel=kernel, groups=groups)
        self.conv3 = Conv_layer(filter_config[1],filter_config[2],stride,actname, True, threeD, pool, actparam=actparam, kernel=kernel, groups=groups)
        self.conv4 = Conv_layer(filter_config[2],filter_config[3],stride,actname, True, threeD, pool, actparam=actparam, kernel=kernel, groups=groups)
        self.conv5 = Conv_layer(filter_config[3],1,               1,     None,    False,threeD, pool,                    kernel=kernel)

        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.drop = nn.Dropout(drop_rate)
        
        self.poollast = poollast
        if self.poollast != "":
            if threeD:
                if self.poollast=='aavg':
                    self.pool_layer = nn.AdaptiveAvgPool3d(1)
                if self.poollast=='amax':
                    self.pool_layer = nn.AdaptiveMaxPool3d(1)
            else:
                if self.poollast=='aavg':
                    self.pool_layer = nn.AdaptiveAvgPool2d(1)
                if self.poollast=='amax':
                    self.pool_layer = nn.AdaptiveMaxPool2d(1)

        # init weights
        if init=="orthogonal":
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d)  :
                  nn.init.orthogonal(m.weight, ortho_gain)
        elif init == "normal":
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d)  :
                    nn.init.normal_(m.weight, mean=0.001, std=200)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    if actname in ['LeakyReLu', 'CELU']:
                        nn.init.kaiming_normal_(m.weight, a=actparam, mode='fan_in', nonlinearity='leaky_relu')
                    if actname in ['SoftPlus']:
                        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    


    def forward(self, x):
        self.out1 = self.conv1(x)
        self.out2 = self.conv2(self.out1)
        self.out3 = self.conv3(self.out2)
        self.out4 = self.conv4(self.out3)
        out5 = self.conv5(self.out4)

        self.feat = [ self.out1, self.out2, self.out3, self.out4]

        if self.drop_rate > 0:
            self.out = self.drop(out5)
        else:
            self.out = out5

        if self.poollast != "":
            return self.pool_layer(self.out)
        else:
            return self.out



############################################################################### 

class Conv_layer(nn.Module):

    """ 
    Convolutionnal layer with batch normalization and activation function
    
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        acitvation (string): acitviation function 
        instanceNorm (boolean): instance normalization use
        3d (boolean): for 3D layers 
   
   """

    def __init__(self, n_in_feat, n_out_feat, stride, activation=None, instanceNorm=False, threeD=True, pool=False, actparam=0.2, kernel=4, groups=1):
        super().__init__()

        if threeD:
            layers = [nn.Conv3d(n_in_feat, n_out_feat, kernel, stride, 1, groups=groups)]
        else:
            layers = [nn.Conv2d(n_in_feat, n_out_feat, kernel, stride, 1, groups=groups)]
        
        if activation=="LeakyReLu":
            layers += [nn.LeakyReLU(0.2,inplace=True)]
        if activation=="SLeakyReLu":
            layers += [SLeakyReLu(eps=1.0,a=actparam,b=1)] 
        if activation=="Sigmoid":
            layers += [torch.nn.Sigmoid()]
        if activation=="SoftPlus":
            layers += [torch.nn.Softplus(beta=actparam, threshold=20)]
        if activation=="CELU":
            layers += [torch.nn.CELU(alpha=actparam, inplace=True)]

        if instanceNorm:
            if threeD:
                layers += [nn.InstanceNorm3d(n_out_feat, momentum=0, affine=True)]
            else:
                layers += [nn.InstanceNorm2d(n_out_feat, momentum=0, affine=True)]

        if threeD: 
            if pool=='avg':
                layers += [nn.AvgPool3d(3, stride=2)]
            if pool=='max':
                layers += [nn.MaxPool3d(3, stride=2)]
            if pool=='aavg':
                layers += [nn.AdaptiveAvgPool3d(1)]
            if pool=='amax':
                layers += [nn.AdaptiveMaxPool3d(1)]
        else:
            if pool=='avg':
                layers += [nn.AvgPool2d(3, stride=2)]
            if pool=='max':
                layers += [nn.MaxPool2d(3, stride=2)]
            if pool=='aavg':
                layers += [nn.AdaptiveAvgPool2d(1)]
            if pool=='amax':
                layers += [nn.AdaptiveMaxPool2d(1)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):

        return self.features(x)


#############################################################################################


class C1Net(nn.Module):
    ''' conv net with only 1x1 conv, norm layer and activation, no down/up-sampling '''
    
    @staticmethod
    def _parse(lstr):
        ''' conv
            lstr: layers list of string [l1,l2,...,lN] 
                  + c12_3_1_inorm  : conv with 12 filters, kernel size 3, dilatation 1, instance norm
            ex: c8_7/c16_3/c8_3/c1_3_1_inorm
        '''
        ltype, ln = [], []
        lks  = [1]*len(lstr)
        ldil = [1]*len(lstr)
        lno  = ['inorm']*(len(lstr)-1) + ['none']
        for i, l in enumerate(lstr): 
            print(l)
            if   l[:1]=='c'  : n,t = 1, int
            else: raise ValueError('error when parsing C1Net string definition')
           
            ltype.append(l[:n])
            ln   .append(t(l[n:].split("_")[0]))
            try:
                lks[i] = int(l[n:].split("_")[1])
            except:
                pass
            try:
                ldil[i] = int(l[n:].split("_")[2])
            except:
                pass
            try:
                lno[i] = l[n:].split("_")[3]
            except:
                pass

        return ltype, ln, lks, ldil, lno


    def __init__(self, dim, ni, ltype, ln, lks, ldil, lno, stride=1, pad=0):
        ''' first layer should be a conv 
            stride is for the first layer only
            ltype[k] in 'c'
            ln[k]    : filters
            lks      : kernel size
            ldil     : dilation
            lno[k]   : normalisation layer
        '''
        super().__init__()

        self.dim = dim

        # if stride>1, first layer should be a conv
        assert( stride==1 or ltype[0]=='c' )

        self.ni = ni
        layers  = []
        nprev   = ni
        for i in range(len(ltype)):
            a = 'none' if (i==len(ltype)-1)    else 'LeakyReLu'

            stride_i = stride if i==0 else 1
            c        =      ConvR (dim=dim, ni=nprev, no=ln[i], ks=lks[i], activation=a, norm=lno[i], stride=stride_i, padding=pad, dilation=ldil[i], padding_mode='replicate') 
            nprev    = ln[i]
            layers.append(c)
        self.no = nprev

        self.convs = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')


    def forward(self, x):
        self.dfeat = [ self.convs(x) ]
        return self.dfeat[0]

##########################################################################################
##########################################################################################
##########################################################################################
# init discriminator

def createDiscParser(parser):
    
    typebool=lambda x:bool(distutils.util.strtobool(x))

    pref1= '-'
    pref2='--'

    parser.add_argument(pref1+'act'        , pref2+'act'        , default="LeakyReLu", type=str, help='LeakyReLu Sigmoid ReLu SoftPlus CELU')
    parser.add_argument(pref1+'actpar'     , pref2+'actpar'     , default=0.2,      type=float, help='activation function parameter')
    parser.add_argument(pref1+'ks'         , pref2+'kernel-size', default=4,        type=int, help='kernel size, default 4)')
    parser.add_argument(pref1+'pool'       , pref2+'pool'       , default="",       type=str, help='pool:  max "" , default "")')
    parser.add_argument(pref1+'poolLast'   , pref2+'poolLast'   , default="",       type=str, help='pool last layer: avg max "" , default "")')
    parser.add_argument(pref1+'drop'       , pref2+'drop'       , default=0.0,      type=float, help='dropout rate , default 0)')
    parser.add_argument(                     pref2+'nfeat'      , default=[64, 128, 256, 512], nargs='+', type=int, help='nb of outpout features for each layers')
    parser.add_argument(                     pref2+'init'       , default='kaiming',type=str,   help='weights init')
    parser.add_argument(                     pref2+'threeD'     , default=True, type=lambda x:bool(distutils.util.strtobool(x)), metavar='BOOL', help='3D')
    parser.add_argument(                     pref2+'nb-features', default=1, type=int, metavar='N', help='number of channel for 2D')
    parser.add_argument(                     pref2+'groups'     , default=1, type=int, metavar='N', help='groups for convolutions')
    
    return parser


def createDisc(args):
    act = {'name' : args.act, 'par' : args.actpar}
    nb_features = args.nb_features
    
    ############################################################
    # moe encoder
    moev               = copy.copy(args.moe)
    moev.moeKLfeat     = args.moeKLfeat
    if moev.beta>0:

        if moev.etype=='I': # identity
            moev.enc = nn.Identity()
        else:
            if   moev.etype=='C': # conv
                moev.enc = ConvR(dim=3 if args.threeD else 2, ni=args.nb_features, no=moev.nfeatmid, ks=3, norm='inorm', activation='None', padding=1)
            elif moev.etype=='CN':
                moev.enc = C1Net(dim=3 if args.threeD else 2, ni=args.nb_features, ltype=moev.ltype, ln=moev.ln, lks=moev.lks, lno=moev.lno, stride=1, pad='same', ldil=moev.ldil)
            else:
                raise ValueError('unknown type for prepro netnetwork in moe: ' + moev.etype) 

            nb_features = moev.nfeatmid

    ############################################################
    #classifier
    disc = CycleGAN_Discriminator(n_init_features=nb_features, threeD=args.threeD, filter_config=args.nfeat, drop_rate=args.drop, pool=args.pool, poollast=args.poolLast, act=act,  kernel=args.kernel_size,  init=args.init, groups=args.groups)
   


    ############################################################
    # moe models
    if args.moe.beta>0:
        return moe.MoE(3 if args.threeD else 2, moev.enc, disc, nullbiasE=True, nullbiasP=True, argmoe=moev)
    else:
        return disc
























avg
