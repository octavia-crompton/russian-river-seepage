import numpy as np
import pickle
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from scipy.ndimage import convolve
import pandas as pd
import matplotlib.colors as colors


from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pylab as plt
import warnings
warnings.filterwarnings("ignore")

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

plt.style.use('ggplot')
import seaborn as sns
sns.set(font_scale = 1.3)

plt.rcParams['axes.facecolor']='w'
plt.rcParams['grid.color']= 'grey'
plt.rcParams['grid.alpha']=0.0
plt.rcParams['axes.linewidth']=0.5
plt.rc('axes',edgecolor='grey')

plt.rcParams['axes.spines.top']= 0
plt.rcParams['axes.spines.right']= 0
plt.rcParams['axes.spines.left']= 1
plt.rcParams['axes.spines.bottom']= 1   

import itertools
from itertools import cycle
lines = ["s","o","d","<"]
linecycler = cycle(lines)

import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'terrain'
