import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import copy
import os

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss



class MultiEarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, k_blocks=1):
        self.patience = patience
        self.verbose = verbose
        self.counter = [0]*k_blocks
        self.best_score = [None]*k_blocks
        self.early_stop = False
        self.early_stop_list = [False]*k_blocks
        self.val_loss_min = np.Inf
        self.delta = delta
        self.k_blocks = k_blocks

    def __call__(self, multi_val_loss, model, path):
        file_name = path + '/' + 'checkpoint.pth'
        if self.best_score == [None]*self.k_blocks:
            state_dict = copy.deepcopy(model.state_dict())
        else:
            print("loading checkpoint")
            state_dict = torch.load(file_name)

        # for each "linear" layer associated with a group (cluster), check if the validation loss is better than the best score
        for layer_j in model.layers_inds.keys():
            j = eval(layer_j) # convert string to int
            layer_j_inds = model.layers_inds[layer_j]
            val_loss = np.mean(multi_val_loss[:,layer_j_inds])
            if self.verbose:
                print(f'inds-{layer_j_inds}-val loss-{val_loss}')
            score_j = -val_loss

            # if the score is None, set it to the current score
            if self.best_score[j] is None:
                self.best_score[j] = score_j
                for l  in state_dict.keys():
                    if f'.{layer_j}.' in l:
                        # check if f'.{layer_j}.' exists more than once
                        if len(l.split(f'.{layer_j}.')) > 2:
                            print("WARNING: more than one occurence of layer_j in state_dict, early stopping may not work as expected")                    
                        state_dict[l] = copy.deepcopy(model.state_dict()[l])

            # if the score is worse than the best score
            elif score_j < self.best_score[j] + self.delta:
                self.counter[j] += 1

                # if time for early stopping, where counter surpasses patience               
                if self.counter[j] >= self.patience:
                    self.early_stop_list[j] = True
                    cur_sd = copy.deepcopy(model.state_dict())
                    for l  in state_dict.keys():
                        if f'.{layer_j}.' in l:
                            # check if f'.{layer_j}.' exists more than once
                            if len(l.split(f'.{layer_j}.')) > 2:
                                print("WARNING: more than one occurence of layer_j in state_dict, early stopping may not work as expected") 
                            if self.verbose:
                                print("setting to best: ", l)
                            cur_sd[l] = copy.deepcopy(state_dict[l])
                    model.load_state_dict(cur_sd)
                    count = 3
                else:
                    count = self.counter[j]
                if self.verbose:
                    print(f'EarlyStopping counter for group-{j}: {count} out of {self.patience}')

            # if the score is better than the best score, update the best score and reset the counter
            else:
                self.best_score[j] = score_j
                for l  in state_dict.keys():
                    if layer_j in l:
                        state_dict[l] = copy.deepcopy(model.state_dict()[l])
                self.counter[j] = 0
                        

        val_loss = multi_val_loss.mean()
        self.save_checkpoint(val_loss, state_dict, path )
        self.early_stop = all(self.early_stop_list)
        return model
    


    def save_checkpoint(self, val_loss, state_dict, path ):
        if self.verbose:
            print(val_loss)
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(state_dict, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))