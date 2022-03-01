# functions for the data exploration

import os
os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")


# Imports ----------------------------

from os import listdir
from os.path import isfile, join
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap
import re
from IPython import display
import glob
import imageio
from datetime import datetime
from tqdm import tqdm
import re
import netCDF4
from pyproj import Proj, transform
import mpl_toolkits

# Variables ----------------------------

reg_dt = '_([^_]+)\.'

# Functions ----------------------------

def get_datetime(filename):
    '''
    Infer datetime from filename
    '''
    try:
        timestamp= re.findall(reg_dt,filename)[0]
        dt = datetime.strptime(timestamp, '%Y%m%d%H%M')
    except:
        print('Error: could not find timestamp in file {}'.format(filename))
    return dt

def load_radar(filename,
               dir_rtcor, rtcor_fbase, dir_aart, aart_fbase,
               as_int=False):
    '''
    Loads the radar file belonging to the filename.
    Looks at the file extension to see from which dataset to retrieve from.
    filename: String filename, can also be just a timestamp (YYYYMMDDhhmm) followed by extension .h5 or .nc
    as_int: if true, the (discreet!) decimal values are converted to integers.
    '''
    path = dir_rtcor + rtcor_fbase + filename
    if filename.endswith('.h5'):
        if rtcor_fbase in filename:
            path = dir_rtcor + filename
        # Open rtcor
        try:
            with h5py.File(path, 'r') as f:
                rain = f['image1']['image_data'][:]
                mask = (rdr == 65535)
                mx = np.ma.masked_array(rain, mask)
                return mx
        except:
            rdr_empty = np.zeros((765, 700, 1))
            return rdr_empty

    if filename.endswith('.nc'):
        path = dir_aart + aart_fbase + filename
        if aart_fbase in filename:
            path = dir_aart + filename
        with netCDF4.Dataset(path, 'r') as f:
            rain = f['image1_image_data'][:]

            if as_int:
                rain *= 100
                rain = rain.astype(int)
            # Change to image format (w,h,c) instead of (c,w,h)
            rain = np.moveaxis(rain, 0, -1)
            return rain

def plot_radar(rdr):
    plt.imshow(np.squeeze(rdr))
    plt.axis('off')
    plt.show()


def get_mask(h5f):
    #obtain binary mask
    mask = np.array(h5f['image2']['image_data'])
    mask[mask==32768] = 1
    mask[mask==65535] = 0
    return mask

def plot_radar(h5file, show_mask=False, fileName=None):
    radar_img = np.array(h5file['image1']['image_data'])
    mask = get_mask(h5file)
    # apply mask to image
    img = radar_img*mask
    plt.imshow(img, vmin=0, vmax=63)
    plt.axis('off')
    if show_mask:
        plt.imshow(mask, cmap='binary', interpolation='none', alpha=0.2)
    if fileName:
        plt.savefig(fileName, bbox_inches='tight')
        display.clear_output(wait=True)
    plt.show()