# Importing essential libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from datetime import date, timedelta, datetime
import pytz
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from io import BytesIO