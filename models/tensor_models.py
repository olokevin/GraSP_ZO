import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_layers.layers import TensorizedLinear_linear, TensorizedLinear_module, TensorizedLinear_module_tonn
from tensor_fwd_bwd.tensorized_linear import TensorizedLinear

class MNIST_FC(nn.Module):
    def __init__(self):
        super(MNIST_FC, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(784, 256, bias=False)
        self.fc2 = nn.Linear(256, 10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(nn.functional.max_pool2d(x, 2))
        x = self.conv2(x)
        x = nn.functional.relu(nn.functional.max_pool2d(x, 2))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        # return nn.functional.log_softmax(x, dim=1)

class MNIST_TTM(nn.Module):
  def __init__(self,tensor_type,max_rank,dropouts=0.1,prior_type='log_uniform',eta=1.0,device=None,dtype=None):
    super(MNIST_TTM, self).__init__()
    self.dropout = nn.Dropout(dropouts)
    # self.shape1 = [[4,7,7,4], [4,8,8,4]]   
    # self.shape2 = [[4,8,8,4], [1,5,2,1]]  
    self.shape1 = [[4,7,7,4], [4,4,4,4]]   
    self.shape2 = [[4,4,4,4], [1,5,2,1]]    
    self.fc1 = TensorizedLinear_module(784, 256, bias=None, shape=self.shape1, tensor_type=tensor_type, max_rank=max_rank,
                                prior_type=prior_type, eta=eta, device=device, dtype=dtype)
    self.fc2 = TensorizedLinear_module(256, 10, bias=None, shape=self.shape2, tensor_type=tensor_type, max_rank=max_rank,
                                prior_type=prior_type, eta=eta, device=device, dtype=dtype)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = torch.flatten(x,1)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x

class MNIST_TT(nn.Module):
    def __init__(self,tensor_type,rank,dropouts=0.2,device=None,dtype=None):
        super(MNIST_TT, self).__init__()
        self.dropout = nn.Dropout(dropouts) 
        self.shape1 = [[4,7,7,4], [4,4,4,4]]   
        self.shape2 = [[4,4,4,4], [1,5,2,1]]

        self.factorization = tensor_type
        self.rank = rank

        self.fc1 = TensorizedLinear(in_tensorized_features=self.shape1[0],
                                    out_tensorized_features=self.shape1[1],
                                    factorization=self.factorization, rank=rank,
                                    bias=None)
        self.fc2 = TensorizedLinear(in_tensorized_features=self.shape2[0],
                                    out_tensorized_features=self.shape2[1],
                                    factorization=self.factorization, rank=rank,
                                    bias=None)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x