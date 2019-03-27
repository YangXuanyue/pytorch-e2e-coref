import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data as tud
from torchvision import models as vmodels
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils
from allennlp.modules.elmo import Elmo, batch_to_ids

import numpy as np
import math
import random

# from scipy.optimize import brentq
# from scipy.interpolate import interp1d
# from sklearn.metrics import roc_curve

random.seed()
np.random.seed()
torch.cuda.seed_all()


def init_params(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)

        if module.bias is not None:
            nn.init.normal_(module.bias.data)

        print('initialized Linear')

    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        print('initialized Conv')

    elif isinstance(module, nn.RNNBase) or isinstance(module, nn.LSTMCell) or isinstance(module, nn.GRUCell):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.normal_(param.data)

        print('initialized LSTM')

    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)
        print('initialized BatchNorm')


def build_len_mask_batch(
        # [batch_size], []
        len_batch, max_len
):
    batch_size, = len_batch.shape
    # [batch_size, max_len]
    idxes_batch = torch.arange(max_len).view(1, -1).repeat(batch_size, 1)
    # [batch_size, max_len] = [batch_size, max_len] >= [batch_size, 1]
    return idxes_batch < len_batch.view(-1, 1)
