from batchcreator_DGMR import minmax
from batchcreator_DGMR import DataGenerator as dg
import config_DGMR as config

import tensorflow as tf
import numpy as np
import h5py
import os
from tqdm import tqdm
import re

output_files = sorted([f for f in os.listdir(config.dir_rtcor_prep)
                       if os.path.isfile(os.path.join(config.dir_rtcor_prep, f))])

def load_h5(file_path):
    radar_img = None
    with h5py.File(file_path, 'r') as f:
        try:
            radar_img = f['image1']['image_data'][:]

            ## Set pixels out of image to 0
            out_of_image = f['image1']['calibration'].attrs['calibration_out_of_image']
            radar_img[radar_img == out_of_image] = 0
            # Sometimes 255 or other number (244) is used for the calibration
            # for out of image values, so also check the first pixel
            radar_img[radar_img == radar_img[0][0]] = 0
            # Original values are in 0.01mm/5min
            # Convert to mm/h:
            radar_img = (radar_img / 100) * 12
        except:
            print("Error: could not read image1 data, file {}".format(file_path))
    return radar_img


def rtcor2npy(in_dir, out_dir, year=None, label_dir=None, preprocess=False, overwrite=False, filenames=None):
    '''
    Preprocess the h5 file into numpy arrays.
    The timestamp, image1 and image2 data of each file is stored
    '''

    add_file_extension = ''
    prefix = ''
    if filenames is not None:
        out_dir = config.dir_rtcor_prep
        # Add file extension to filename
        # Add a prefix to filename
        add_file_extension = '.h5'
        prefix = config.prefix_rtcor
    else:
        d = in_dir + str(year)
        filenames = []
        for m in os.listdir(d):
            dir_m = os.path.join(d, m)
            if os.path.isdir(dir_m):
                for f in os.listdir(dir_m):
                    if f.endswith('.h5') and f.startswith(config.prefix_rtcor + str(year)):
                        filenames.append(f)
        filenames = sorted(filenames)

        # Create directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if label_dir and not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Pattern for detecting timestamp in filename
    regex_file = re.compile('(\d{12})\.h5')

    for filename in tqdm(filenames):
        filename = filename + add_file_extension

        timestamp = regex_file.findall(filename)[0]
        scan_fn = out_dir + '/' + "{}.npy".format(timestamp)
        year = timestamp[0:4]
        path_scan = in_dir + str(year) + '/' + prefix + filename

        if not overwrite and timestamp + '.npy' in output_files:
            # Skip this file if already processed,
            # go to next file in list
            continue
        try:
            radar_img = load_h5(path_scan)
            if preprocess:
                radar_img = perform_preprocessing(radar_img)
            np.save(scan_fn, radar_img)


        except Exception as e:
            print(e)
            # np.save(scan_fn, radar_data)

def perform_preprocessing(x, downscale256=True):
    x = minmax(x, norm_method='minmax', undo=False, convert_to_dbz = True)
    x = np.expand_dims(x, axis=-1)
    if downscale256:
        # First make the images square size
        x = dg.pad_along_axis(dg, x, axis=0, pad_size=3)
        x = dg.pad_along_axis(dg, x, axis=1, pad_size=68)
        x =  tf.image.resize(x, (256, 256))
    return x
