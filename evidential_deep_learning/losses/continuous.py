import torch
import torch.nn as nn
import numpy as np

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)
    #print("nll:", nll)
    return torch.mean(nll) if reduce else nll  


def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    v1, v2, a1, a2, b1, b2 = torch.tensor(v1), torch.tensor(v2), torch.tensor(a1), torch.tensor(a2), torch.tensor(b1), torch.tensor(b2)
    KL = 0.5 * (a1 - 1) / b1 * (v2 * torch.square(mu2 - mu1)) \
        + 0.5 * v2 / v1 \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 + a2 * torch.log(b1 / b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1) \
        - (b1 - b2) * a1 / b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.002, reduce=True, kl=False):
    error = torch.abs(y-gamma)
    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi
    #print("Reg:", reg)
    return torch.mean(reg) if reduce else reg

def EvidentialRegression(y_true, evidential_output, coeff=0.2):#0.2
    gamma, v, alpha, beta = torch.split(evidential_output, int(evidential_output.shape[-1]/4), dim=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg
