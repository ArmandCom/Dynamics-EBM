import torch
import torch.nn as nn
import scipy.special as sp
import numpy as np
from scipy.linalg import fractional_matrix_power

class SoS_loss(nn.Module):
    def __init__(self, d, mord=1, gpu_id=None, num_samples = None):
        self.gpu = gpu_id
        d = 1
        super(SoS_loss, self).__init__()
        self.mord = mord
        s = int(sp.binom(mord+d,mord))
        self.s = s
        if num_samples is None:
            num_samples = s
        # Ad = torch.zeros((s,s), device = DEVICE, requires_grad=False)
        # for i in range(s):
        #     Ad[i][s-i-1] = 1
        # self.Ad = Ad
        # You will need to calculate power parameter, you don't need the stuff above
        powers = []
        for i in range(2,mord+1):
            powers.append(torch.from_numpy(self.exponent(i,d)).type(torch.FloatTensor)[None].permute(2, 1, 0))
        self.powers = powers
        self.ones = torch.ones((1, 1, num_samples * 10), device = self.gpu, requires_grad=False)
        # TODO: check if this is correct. It might not be because ones oddly depends on the batchsize.
        # self.register_buffer('powers', powers)

    def exponent(self, n, K):
        # Credit: python implementation from the original MATLAb code of Rene Vidal, 2013
        idd = np.eye(K)
        exp = idd
        for i in range(2,n+1):
            rene = []
            for j in range(K):
                for k in range(exp.shape[0]-int(sp.binom(i+K-j-2,i-1)), exp.shape[0]):
                    rene.append(idd[j]+exp[k])
            exp = np.array(rene)
        return exp   

    def calcQ(self, V, x, Minv=None, kernelized=True):
        #  x: row vectors in veronese space
        # V is s-by-n where n is number of samples s is dimension
        x = x.permute(0, 2, 1)

        if kernelized:
            #  This function calculates Q values usinhg kernelized version if kernelized is true
            if Minv is None:
                Minv = V @ \
                torch.inverse(self.rho_val(V)*
                              torch.eye(V.shape[2], device = V.device, requires_grad=False)[None]
                              + V.permute(0, 2, 1) @ V) @ V.permute(0, 2, 1)
            Q = torch.diagonal(x @ x.permute(0, 2, 1) - x @ Minv @ x.permute(0, 2, 1), dim1=-1, dim2=-2)
        else:
            if Minv is None:
                Minv = torch.inverse(((V @ V.permute(0, 2, 1)) / V.shape[2]))
            Q = torch.diagonal(x @ Minv @ x.permute(0, 2, 1), dim1=-1, dim2=-2) # Note: to test
        return Q

    def calcMinv(self, V, kernel=False):
        if kernel:
            return V @ \
                        torch.inverse(self.rho_val(V)*
                                      torch.eye(V.shape[2], device = V.device, requires_grad=False)[None]
                                      + V.permute(0, 2, 1) @ V) @ V.permute(0, 2, 1)
        else: return torch.inverse(((V @ V.permute(0, 2, 1)) / V.shape[2]))

    def veronese(self, X, n, powers=None):
        if n==0:
            if self.ones.shape[0] < X.shape[0]: self.ones = self.ones[:1].repeat_interleave(X.shape[0], dim=0)
            y = self.ones[:X.shape[0], :, :X.shape[2]] # device = self.gpu,
            if y.device != X.device:
                y = y.to(X.device)
        elif n==1:
            y = X
        else:
            if X.min() < 0:
                raise ValueError("Negative values not allowed")
            if powers.any()==None:
                raise ValueError("powers cannot be None for mord>=2")
            X = torch.clamp(X, min=1e-10)
            y = torch.exp( powers
                           @ torch.log(X)) # Note: Method to accelerate computation. Log doesn't take negative values.
            # TODO: Check gradients here. log(X) might be a problem near 0.

        return y     

    def vmap(self, X):
        X = X[None].permute(2,1,0) # Testing feature, remove
        #  veronese map function
        vx = torch.cat((self.veronese(X.permute(0, 2, 1), 0), self.veronese(X.permute(0, 2, 1),1)),1)

        p = 0
        if self.powers[0].device != X.device:
            self.powers = [self.powers[pi].to(X.device) for pi in range(len(self.powers))]
        for i in range(2,self.mord+1):
            vx = torch.cat((vx, self.veronese(X.permute(0, 2, 1), i, self.powers[p])),1)
            p+=1
        return vx    

    def rho_val(self, V):
        import math
        return torch.norm( V.permute(0, 2, 1) @ V )/(500*math.sqrt(V.shape[2]))

    # def mom(self, X):
    #     # Main moment method which calculates moment matrix
    #     Vx = self.vmap(X)
    #     rho = self.rho_val(Vx)
    #     # Commented out below is the secondary formulation for moments
    #     # Mx = rho * torch.eye(self.s, device = self.gpu, requires_grad=False) + (Vx @ torch.t(Vx)) / Vx.shape[1]
    #     Mx = (Vx @ torch.t(Vx)) / Vx.shape[1]
    #     return Mx
 
    # def forward(self, Ms, Mt, U, source_tr, target_tr, label_source, use_squeeze = True):
    #     # You probably don't need this, this was specific to my project
    #
    #     Vsr = self.vmap(source_tr)
    #     rho = self.rho_val(Vsr)
    #
    #     Mth = torch.tensor( np.real( fractional_matrix_power(Mt.detach().cpu().numpy(), -0.5) ) , requires_grad=True, device=self.gpu).float()
    #     Msh = torch.tensor( np.real( fractional_matrix_power(Ms.detach().cpu().numpy(), 0.5) ) , requires_grad=True, device=self.gpu).float()
    #
    #     Vtr = self.vmap(target_tr)
    #
    #
    #     A = Msh @ U @ Mth
    #     reg_loss = (U @ U.t() - torch.eye(U.shape[0], device = self.gpu, requires_grad=False)).pow(2).sum()
    #     # print("Error on orthogonality: ", reg_loss)
    #     trloss = torch.sum(self.calcQ( Vsr, torch.t(A @ Vtr), rho)) + reg_loss
    #
    #     if not use_squeeze:
    #         sqloss = 0
    #     else:
    #         # TODO
    #         inliers = torch.sum(self.calcQ( Vsr, torch.t(Vsr), rho))
    #         sqloss = inliers
    #         # Vsource = self.vmap(source[label_source[1]!=label_source[0]][:])
    #         # outliers = torch.sum(self.calcQ( Vsr, torch.t(Vsource), rho))
    #         # sqloss = inliers - outliers
    #     return trloss, sqloss
