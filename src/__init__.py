"""
EEG Motor Imagery Classification Package

A deep learning system for classifying motor imagery from EEG brain signals.
Implements EEGNet, CNN-LSTM, and Transformer architectures for BCI applications.
"""

__version__ = "1.0.0"
__author__ = "Sanober"

from . import config
from . import data_loader
from . import preprocessing
from . import training
from . import evaluation
from . import visualization
