import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

class Mine(nn.Module):
    def __init__(self, fmaps=16, hidden_size=100, dimA=1, dimB=1, lr=1e-3, init_scale=0.02, fdiv=False):
        super().__init__()
        self.fmaps = fmaps
        self.hidden_size = hidden_size
        self.flattened_dim = None
        if type(dimA) == int:
            self.fc1 = nn.Linear(dimA+dimB, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.fc1.weight,std=init_scale)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight,std=init_scale)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight,std=init_scale)
            nn.init.constant_(self.fc3.bias, 0)
            nn.init.normal_(self.fc4.weight,std=init_scale)
            nn.init.constant_(self.fc4.bias, 0)
        else:
            self.dimA = dimA
            self.dimB = dimB
            ca, ha, wa = dimA
            cb, hb, wb = dimB
            
            ksize = 5
            self.fc1a = nn.Sequential(nn.BatchNorm2d(ca), nn.Conv2d(ca , fmaps, ksize), nn.AvgPool2d(2), nn.ELU(), nn.Conv2d(fmaps, fmaps, ksize), nn.ELU(), nn.Conv2d(fmaps, fmaps, 5))
            self.fc1b = nn.Sequential(nn.BatchNorm2d(cb), nn.Conv2d(cb, fmaps, ksize), nn.AvgPool2d(2), nn.ELU(), nn.Conv2d(fmaps, fmaps, ksize), nn.ELU(), nn.Conv2d(fmaps, fmaps, 5))
            self.fc2a = nn.Sequential(nn.Conv2d(fmaps, fmaps, 5), nn.AvgPool2d(2), nn.ELU(), nn.Conv2d(fmaps, fmaps, 5), nn.ELU(), nn.Conv2d(fmaps, fmaps, 5), nn.BatchNorm2d(fmaps))
            self.fc2b = nn.Sequential(nn.Conv2d(fmaps, fmaps, 5), nn.AvgPool2d(2), nn.ELU(), nn.Conv2d(fmaps, fmaps, 5),  nn.ELU(), nn.Conv2d(fmaps, fmaps, 5), nn.BatchNorm2d(fmaps)) # 96
            
            self.outputA = self.fc2a(self.fc1a(torch.ones((1, ca, ha, wa))))
            self.flattened_dim = 2 * self.outputA.view(1, -1).shape[1]
            self.flattened_dim = fmaps #flattened_dim # 2592
            self.fc3 = nn.Linear(self.flattened_dim, hidden_size)
            self.fc4 = nn.Linear(hidden_size, 1)
            
        self.mine_net_optim = optim.Adam(self.parameters(), lr=lr)
        self.ma_et = 1.
        self.ma_rate=0.01
        self.dimA = dimA
        self.dimB = dimB
        self.fdiv = fdiv
        
    def __repr__(self):
        return f"Mine(fmaps={self.fmaps}, hidden_size={self.hidden_size}, flattened_dim={self.flattened_dim}, latent_spatial={self.outputA.shape})[{np.sum([p.numel() for p in self.parameters() if p.requires_grad])}]"
        
    def _forward(self, input):
        input = input.to(next(self.parameters()).device)
        A, B = input[:, :self.dimA[0]], input[:, self.dimA[0]:]
        
        outputA = F.elu(self.fc1a(A))
        outputA = F.elu(self.fc2a(outputA))
        
        outputB = F.elu(self.fc1b(B))
        outputB = F.elu(self.fc2b(outputB))
        
        output = (outputA * outputB).mean((2,3))
        
        output = F.elu(self.fc3(output))
        output = self.fc4(output)
        return output.cpu()
    
    def mutual_information(self, joint, marginal):
        mi_lb , j, et, m = self(joint, marginal)
        self.ma_et = ((1-self.ma_rate)*self.ma_et + self.ma_rate*torch.mean(et)).detach()
        
        # unbiasing use moving average
        # loss = -(torch.mean(t) - (1/self.ma_et)*torch.mean(et))
        
        # replacing by Binary CE optimization
        acc = ((j > 0).sum() + (m < 0).sum())/(len(j)+len(m))
        loss = torch.nn.BCEWithLogitsLoss()(torch.cat([j, m]).flatten(), torch.cat([torch.ones(len(j)), torch.zeros(len(m))]))
        return loss, mi_lb, acc, torch.cat([j, m]).view(-1, 1).detach()
        
    def forward(self, joint, marginal):
        j = self._forward(joint)
        m = self._forward(marginal)
        if self.fdiv:
            et = torch.exp(m - 1)
        else:
            et = torch.exp(m)
        mi_lb = torch.mean(j) - torch.log(torch.mean(et))
        return mi_lb, j, et, m
    
    def step(self, joint , marginal):
        mi_lb , t, et, m = self(joint, marginal)
        self.ma_et = ((1-self.ma_rate)*self.ma_et + self.ma_rate*torch.mean(et)).detach()
        # unbiasing use moving average
        loss = -(torch.mean(t) - (1/self.ma_et)*torch.mean(et))
        # use biased estimator
        # loss = - mi_lb
        loss.backward()
        self.mine_net_optim.step()
        self.mine_net_optim.zero_grad()
        return mi_lb
    
    
    @staticmethod
    def sample_batch(A, B, batch_size=100, sample_mode='joint'):
            if sample_mode == 'joint':
                index = np.random.choice(range(A.shape[0]), size=batch_size, replace=False)
                batch = torch.cat([A[index], B[index]], axis=1)
            else:
                joint_index = np.random.choice(range(A.shape[0]), size=batch_size, replace=False)
                marginal_index = np.random.choice(range(A.shape[0]), size=batch_size, replace=False)
                batch = torch.cat([A[joint_index], B[marginal_index]],axis=1)
            return batch     
    
    def fit(self, data, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3), validation_ratio=10):
        
        # data is x or y
        result = list()
        device = next(self.parameters()).device

        data_idx = np.random.permutation(range(len(data)))
        val_idx = data_idx[:len(data_idx)//validation_ratio]
        dat_idx = data_idx[len(data_idx)//validation_ratio:]
       
        for i in range(iter_num):
            batch_joint = Mine.sample_batch(data[dat_idx],batch_size=batch_size, dimA=self.dimA, dimB=self.dimB)
            batch_marginal = Mine.sample_batch(data[dat_idx],batch_size=batch_size,sample_mode='marginal', dimA=self.dimA, dimB=self.dimB)
            
            mi_lb = self.step(torch.FloatTensor(batch_joint).to(device), torch.FloatTensor(batch_marginal).to(device))
            
            trn_loss = mi_lb.detach().cpu().numpy()
            
            with torch.no_grad():
                batch_joint = Mine.sample_batch(data[val_idx],batch_size=len(val_idx), dimA=self.dimA, dimB=self.dimB)
                batch_marginal = Mine.sample_batch(data[val_idx],batch_size=len(val_idx),sample_mode='marginal', dimA=self.dimA, dimB=self.dimB)
                mi_lb , t, et = self.mutual_information(torch.FloatTensor(batch_joint).to(device), torch.FloatTensor(batch_marginal).to(device))
                val_loss = mi_lb.detach().cpu().numpy()
            
            result.append((trn_loss, val_loss))
            if (i+1)%(log_freq)==0:
                print(np.asarray(result[-20:]).mean())
        return np.asarray(result)