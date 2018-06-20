import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Function, gradient_check
from chainer import Variable, optimizer, serializer, utils
