from .. import fdtd
from ..misc import NOT_PARALLEL

import numpy as np
import torch

__author__ = "Sean Hooten"
__copyright__ = 'Copyright 2023 Hewlett Packard Enterprise Development LP.'
__license__ = "BSD-3"
__maintainer__ = "Sean Hooten"
__status__ = "development"

class FDTD(fdtd.FDTD):
    def __init__(self, *args, **kwargs):
        super(FDTD, self).__init__(*args, **kwargs)

    def calc_ydAx_topology(self, domain, update_mu, sig_eps=None, sig_mu=None, planar=False, lam=None):
        # NOTE: Assumes isotropic materials

        Ex = self.get_field('Ex', domain)
        Ex_adj = self.get_adjoint_field('Ex', domain)
        Ey = self.get_field('Ey', domain)
        Ey_adj = self.get_adjoint_field('Ey', domain)
        Ez = self.get_field('Ez', domain)
        Ez_adj = self.get_adjoint_field('Ez', domain)

        #if sig_eps is not None:
        #    #grad_eps = 2*sig_eps[1]*np.imag(sig_eps[0] * (1.0 - sig_eps[0]) * (Ex*Ex_adj + Ey*Ey_adj + Ez*Ez_adj))
        #    mult = sig_eps[1] * sig_eps[0] * (1.0 - sig_eps[0])
        #else:
        #    #grad_eps = 2*np.imag(Ex*Ex_adj + Ey*Ey_adj + Ez*Ez_adj)
        #    mult = 1

        if planar:
            grad_eps = 2 * sig_eps * np.imag(np.sum(Ex * Ex_adj, axis=0) + np.sum(Ey * Ey_adj, axis=0) + np.sum(Ez * Ez_adj, axis=0))
        else:
            grad_eps = 2 * sig_eps * np.imag(Ex * Ex_adj + Ey * Ey_adj + Ez * Ez_adj)

        if update_mu:
            Hx = self.get_field('Hx', domain)
            Hx_adj = self.get_adjoint_field('Hx', domain)
            Hy = self.get_field('Hy', domain)
            Hy_adj = self.get_adjoint_field('Hy', domain)
            Hz = self.get_field('Hz', domain)
            Hz_adj = self.get_adjoint_field('Hz', domain)

            if planar:
                grad_mu = -2 * sig_mu * np.imag(np.sum(Hx * Hx_adj, axis=0) + np.sum(Hy * Hy_adj, axis=0) + np.sum(Hz * Hz_adj, axis=0))
            else:
                grad_mu = -2 * sig_mu * np.imag(Hx * Hx_adj + Hy * Hy_adj + Hz * Hz_adj)

            #if sig_mu is not None:
            #    grad_mu = -2*sig_mu[1]*np.imag(sig_mu[0] * (1.0 - sig_mu[0]) * (Hx*Hx_adj + Hy*Hy_adj + Hz*Hz_adj))
            #else:
            #    grad_mu = -2*np.imag(Hx*Hx_adj + Hy*Hy_adj + Hz*Hz_adj)

            grad = np.concatenate([grad_eps.ravel(), grad_mu.ravel()], axis=0)
        else:
            grad = grad_eps.ravel()

        return grad

    def calc_ydAx_autograd(self, domain, update_mu, eps_autograd, mu_autograd):
        Ex = torch.as_tensor(self.get_field('Ex', domain))
        Ex_adj = torch.as_tensor(self.get_adjoint_field('Ex', domain))
        Ey = torch.as_tensor(self.get_field('Ey', domain))
        Ey_adj = torch.as_tensor(self.get_adjoint_field('Ey', domain))
        Ez = torch.as_tensor(self.get_field('Ez', domain))
        Ez_adj = torch.as_tensor(self.get_adjoint_field('Ez', domain))
        pseudoloss = eps_autograd[0] * Ex * Ex_adj + \
                     eps_autograd[1] * Ey * Ey_adj + \
                     eps_autograd[2] * Ez * Ez_adj
        if update_mu:
            Hx = torch.as_tensor(self.get_field('Hx', domain))
            Hx_adj = torch.as_tensor(self.get_adjoint_field('Hx', domain))
            Hy = torch.as_tensor(self.get_field('Hy', domain))
            Hy_adj = torch.as_tensor(self.get_adjoint_field('Hy', domain))
            Hz = torch.as_tensor(self.get_field('Hz', domain))
            Hz_adj = torch.as_tensor(self.get_adjoint_field('Hz', domain))
            pseudoloss = pseudoloss - mu_autograd[0] * Hx * Hx_adj - \
                                      mu_autograd[1] * Hy * Hy_adj - \
                                      mu_autograd[2] * Hz * Hz_adj
        if NOT_PARALLEL:
            return 2 * pseudoloss.imag.sum()
        else:
            return
