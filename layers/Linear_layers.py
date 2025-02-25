import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN




class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean    
    

class DLinear(nn.Module):
    def __init__(self, input_len, output_len, kernel_size = 25):
        super(DLinear, self).__init__()
        self.seasonal = nn.Linear(input_len, output_len)
        self.trend = nn.Linear(input_len, output_len)
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.decompsition = series_decomp(kernel_size)

    def forward(self, x):
        # x: [Batch, Input length,Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_output = self.seasonal(seasonal_init.permute(0,2,1)).permute(0,2,1)
        trend_output = self.trend(trend_init.permute(0,2,1)).permute(0,2,1)
        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]    
    
class Linear(nn.Module):
    def __init__(self, input_len, output_len):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x):
        # x: [Batch, Input length,Channel]
        x = x.clone()
        x = self.linear(x.permute(0,2,1)).permute(0,2,1).clone()
        return x # to [Batch, Output length, Channel]   
    

class NLinear(nn.Module):
    def __init__(self, input_len, output_len):
        super(NLinear, self).__init__()
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x):
        # x: [Batch, Input length,Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = self.linear(x.permute(0,2,1))
        return x.permute(0,2,1)+seq_last # to [Batch, Output length, Channel]   
    
 
class RLinear(nn.Module):
    def __init__(self, input_len, output_len):
        super(RLinear, self).__init__()
        self.linear = nn.Linear(input_len, output_len)
        self.revin_layer = RevIN(num_features = None, affine=False, norm_type = None, subtract_last = False)

    def forward(self, x):
        # x: [Batch, Input length,Channel]
        x = x.clone()
        x = self.revin_layer(x, 'norm')
        x = self.linear(x.permute(0,2,1)).permute(0,2,1).clone()
        x = self.revin_layer(x, 'denorm')
        return x # to [Batch, Output length, Channel]   
    
