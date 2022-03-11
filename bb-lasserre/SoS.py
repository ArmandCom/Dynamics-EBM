import torch
import torch.nn as nn
import scipy.special as sp
import numpy as np
from scipy.linalg import fractional_matrix_power

class SoS_loss(nn.Module):
    def __init__(self, d, mord=1, gpu_id=0):
        DEVICE = torch.device('cuda:%d'%gpu_id if torch.cuda.is_available() else 'cpu')
        self.gpu = DEVICE
        super(SoS_loss, self).__init__()
        self.mord = mord
        s = int(sp.binom(mord+d,mord))
        self.s = s
        Ad = torch.zeros((s,s), device = DEVICE, requires_grad=False)
        for i in range(s):
            Ad[i][s-i-1] = 1
        self.Ad = Ad
        # You will need to calculate power parameter, you don't need the stuff above
        powers = []
        for i in range(2,mord+1):
            powers.append(self.exponent(i,d))
        self.powers = powers
        

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

    def calcQ(self, V, x, rho):
        #  x: row vectors in veronese space
        # V is s-by-n where n is number of samples s is dimension
        #  This function calculates Q values usinhg kernelized version
        Q = torch.diag(x @ torch.t(x) - x @ V @ \
        torch.inverse(rho*torch.eye(V.shape[1], device = self.gpu, requires_grad=False) + torch.t(V) @ V) \
        @ torch.t(V) @ torch.t(x))
        return Q

    def veronese(self, X, n, powers=None):
        if n==0:
            y = torch.ones((1,X.shape[1]), device = self.gpu, requires_grad=False)
        elif n==1:
            y = X
        else:
            if powers.any()==None:
                raise ValueError("powers cannot be None for mord>=2")
            X[torch.abs(X)<1e-10] = 1e-10
            y = torch.exp(torch.from_numpy(powers).to(self.gpu).type(torch.cuda.FloatTensor) @ torch.log(X))

        return y     

    def vmap(self, X):
        #  veronese map function
        vx = torch.cat((self.veronese(torch.t(X), 0), self.veronese(torch.t(X),1)),0)
        # dtype = torch.cuda.FloatTensor
        # vx = torch.cat((self.veronese(torch.t(X), 0).type(dtype), self.veronese(torch.t(X),1).type(dtype)),0)

        p = 0
        for i in range(2,self.mord+1):
            vx = torch.cat((vx, self.veronese(torch.t(X), i, self.powers[p])),0)
            p+=1
        return vx    

    def rho_val(self, V):
        import math
        return torch.norm( torch.t(V) @ V )/(500*math.sqrt(V.shape[1]))

    def mom(self, X):
        # Main moment method which calculates moment matrix
        Vx = self.vmap(X)
        rho = self.rho_val(Vx)
        # Commented out below is the secondary formulation for moments
        # Mx = rho * torch.eye(self.s, device = self.gpu, requires_grad=False) + (Vx @ torch.t(Vx)) / Vx.shape[1]
        Mx = (Vx @ torch.t(Vx)) / Vx.shape[1]

        return Mx

 
    def forward(self, Ms, Mt, U, source_tr, target_tr, label_source, use_squeeze = True):
        # You probably don't need this, this was specific to my project
        
        Vsr = self.vmap(source_tr)
        rho = self.rho_val(Vsr)
    
        Mth = torch.tensor( np.real( fractional_matrix_power(Mt.detach().cpu().numpy(), -0.5) ) , requires_grad=True, device=self.gpu).float()
        Msh = torch.tensor( np.real( fractional_matrix_power(Ms.detach().cpu().numpy(), 0.5) ) , requires_grad=True, device=self.gpu).float()

        Vtr = self.vmap(target_tr)
      

        A = Msh @ U @ Mth
        reg_loss = (U @ U.t() - torch.eye(U.shape[0], device = self.gpu, requires_grad=False)).pow(2).sum()
        # print("Error on orthogonality: ", reg_loss)
        trloss = torch.sum(self.calcQ( Vsr, torch.t(A @ Vtr), rho)) + reg_loss

        if not use_squeeze:
            sqloss = 0
        else:
            # TODO
            inliers = torch.sum(self.calcQ( Vsr, torch.t(Vsr), rho))
            sqloss = inliers
            # Vsource = self.vmap(source[label_source[1]!=label_source[0]][:])
            # outliers = torch.sum(self.calcQ( Vsr, torch.t(Vsource), rho))
            # sqloss = inliers - outliers
        return trloss, sqloss
