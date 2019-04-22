import configs
from model import Model
from model_utils import *
import os
from log import log
import time
import data_utils
import re
from itertools import chain
from allennlp.training.optimizers import DenseSparseAdam
import metrics
import traceback
from runner import Runner

runner = Runner()
runner.test_gpu()