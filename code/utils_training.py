import pickle
import random
from re import S
import warnings
from matplotlib import pyplot as plt
import numpy as np
from sympy import Li
import torch
from torch_geometric.nn import Linear, GCNConv, SAGEConv
from torch.nn import ReLU, Sigmoid
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# warnings.filterwarnings('ignore')

def __seed_all(seed):
    '''
    Set random seeds for reproducability
    '''
    if not seed:
        seed = 951201
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#########################################################################################################


class ScheduledOptim():
    def __init__(self, optimizer, n_warmup_steps, decay_rate, steps=None):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.decay = decay_rate
        self.n_steps = 0
        self.steps = steps
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_lr = optimizer.param_groups[0]['lr']

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        # self._optimizer.step()
        self.update()
    
    def get_lr(self):
        return self.current_lr
    
    def update(self):
        """
        Update the learning rate.

        - During the first self.n_warmup_steps, gradually increase the learning rate from 0 to the self.initial_lr.
        - Then, decay the learning rate with the given probability self.decay at each decay steps self.steps.
        - If decay steps are not given, decay the learning rate on every epoch.
        """
        if self.n_steps <= self.n_warmup_steps:
              lr = self.n_steps / self.n_warmup_steps * self.initial_lr
        elif self.n_steps == self.n_warmup_steps:
              lr = self.initial_lr
        else:
            if self.steps is None:
                lr = self.initial_lr * self.decay
            else:
                if self.n_steps in self.steps:
                      lr = self.current_lr * self.decay
                else:
                      lr = self.current_lr
          
        self.current_lr = lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.current_lr

        self.n_steps += 1

#########################################################################################################

class GNN(torch.nn.Module):
    '''
    Graph Neural Network
    '''
    def __init__(self):
        super(GNN, self).__init__()

        self.sage1 = SAGEConv(17,1,improved=True)
        # self.gconv2 = GCNConv(1,1)
        # self.gconv3 = GCNConv(32,64)
        # self.gconv4 = GCNConv(64,32)
        # self.gconv5 = GCNConv(32,16)
        # self.gconv6 = GCNConv(16,16)
        # self.sigmoid1 = F.relu()

        self.out = Linear(1,1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = F.elu(self.sage1(x, edge_index))
        # x = F.dropout(x, 0.8, training=self.training)
        # x = F.relu(self.gconv2(x, edge_index, edge_weight))
        # x = F.dropout(x, 0.9, training=self.training)
        # x = F.relu(self.gconv3(x, edge_index, edge_weight))
        # x = F.dropout(x, 0.9, training=self.training)
        # x = F.relu(self.gconv4(x, edge_index, edge_weight))
        # x = F.dropout(x, 0.9, training=self.training)
        # x = F.relu(self.gconv5(x, edge_index, edge_weight))
        # x = F.dropout(x, 0.9, training=self.training)
        # x = F.relu(self.gconv6(x, edge_index, edge_weight))
        # x = F.dropout(x, 0.9, training=self.training)

        x = self.out(x)

        return x
    
#########################################################################################################

def getLoader(datalist_dir, batch_fraction):
    
    with open(datalist_dir, 'rb') as f:
        datalist = pickle.load(f)

    batch_size = int(len(datalist)*batch_fraction)

    # g = torch.Generator()
    # g.manual_seed(seed)

    loader = DataLoader(datalist, batch_size=batch_size, shuffle=False)

    return loader

#########################################################################################################

def train(model, train_params, train_loader, val_loader, logfile_dir, logfig_dir):

    # __seed_all(seed)
    
    opt_name = train_params['opt_name']
    n_epoch = train_params['n_epoch']
    lr = train_params['lr']
    weight_decay = train_params['weight_decay']
    loss_fname = train_params['loss_fname']
    lr_decay_rate = train_params['lr_decay_rate']

    if opt_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr= lr, betas=(0.9,0.999),eps=1e-08,weight_decay=weight_decay)
    
    scheduler = ScheduledOptim(optimizer, n_warmup_steps=100, decay_rate=lr_decay_rate)
    scheduler.update()
    
    if loss_fname == 'mseLoss':
        loss_func = F.mse_loss

    log_f = open(logfile_dir, 'w')
    print(f'opt_name,n_epoch,lr,weight_decay,loss_func', file=log_f, flush=True)
    print(f'{opt_name},{n_epoch},{lr},{weight_decay},{loss_fname}', file=log_f, flush=True)
    print(f'epoch, train_loss, val_loss, val_meanARE', file=log_f, flush=True)

    fname, ext = logfig_dir.split('.')
    for epoch in range(n_epoch):
        model.train()
        for train_batch in train_loader:
            optimizer.zero_grad()

            train_pred = model(train_batch).squeeze()
            train_true = train_batch.fip.squeeze()

            train_loss = loss_func(train_pred, train_true)

            train_loss.backward()

            optimizer.step()

        scheduler.step()

        val_batch = next(iter(val_loader))
        val_pred = model(val_batch)
        val_true = val_batch.fip
        val_loss = loss_func(val_pred, val_true)
        val_meanARE = torch.mean(torch.abs((val_pred - val_true)/val_true))    

        
        
        if epoch%5 == 0:
            print("[Epoch {}]".format(epoch))
            print("[Loss : {}]".format(val_loss.item()))
            print("[meanARE : {}]".format(val_meanARE))
            print("[Learning rate : {}]".format(optimizer.param_groups[0]['lr']) )



    log_f.close()