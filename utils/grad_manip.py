import torch
import numpy as np

class MagPenalty:
    def __init__(self, penalty_param):
        self.penalty_param = penalty_param
        
    def get_weights(self, objectives_v = None, objectives_h = None):
        if objectives_v is not None:
            objectives_v = objectives_v.flatten().detach()
            # if element in objectives_v is zero set to 1 to avoid division by zero.
            # this does not affect the result since the corresponding loss will be zero in any case
            d_objectives_v = torch.where(objectives_v==0,torch.tensor(1.),objectives_v)
            self.loss_w_v = (1/d_objectives_v)**self.penalty_param

        if objectives_h is not None:
            objectives_h = objectives_h.flatten().detach()
            # if element in objectives_h is zero set to 1 to avoid division by zero.
            # this does not affect the result since the corresponding loss will be zero in any case
            d_objectives_h = torch.where(objectives_h==0,torch.tensor(1.),objectives_h)
            self.loss_w_h = (1/d_objectives_h)**self.penalty_param

        if objectives_v is not None and objectives_h is not None:
            self.loss_w = self.loss_w_h.view(-1,1)  @ self.loss_w_v.view(1,-1) 
        elif objectives_v is not None:
            self.loss_w = self.loss_w_v
        elif objectives_h is not None:
            self.loss_w = self.loss_w_h
        
        return self.loss_w  
