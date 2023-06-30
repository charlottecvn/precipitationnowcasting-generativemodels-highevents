import config_DGMR
import numpy as np
import os
from tqdm import tqdm
import h5py

threshold_intensity = 30
year = "2019"

print("Start computing rain intensities for year : ", year, ", and threshold = ", threshold_intensity)

def rain_intensity (img):
    '''
    Computes the rain intensity of an image, using to the dBZ and dBR
    The function leads to a numpy array with the intensities
    '''
    b = 1.56 #20
    a = 58.53 #20
    dBZ = img * 70.0 - 10.0
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    return np.power(10, dBR / 10.0)
    
def load_h5(path):
    '''
    The orginial input images are stored in .h5 files.
    This function loads them and converts them to numpy arrays
    '''
    radar_img = None
    with h5py.File(path, 'r') as f:
        try:
            radar_img = f['image1']['image_data'][:]

            ## Set pixels out of image to 0
            out_of_image = f['image1']['calibration'].attrs['calibration_out_of_image']
            radar_img[radar_img == out_of_image] = 0
            # Sometimes 255 or other number (244) is used for the calibration
            # for out of image values, so also check the first pixel
            radar_img[radar_img == radar_img[0][0]] = 0
        except:
            print("Error: could not read image1 data, file {}".format(path))
    return radar_img
    
    
radar_dir = config_DGMR.dir_rtcor
label_dir = config_DGMR.dir_labels

root = radar_dir + year
print(root)

files = sorted([name for path, subdirs, files in os.walk(root) for name in files])
files = files[1:]

cluttermask = ~np.load(config_DGMR.path_code + 'cluttermask.npy')

print("Number of files from ", year, " : ", len(files))

count_print = 0
print(count_print)

for f in tqdm(files):
    print(f)
    ts = f.replace(config_DGMR.prefix_rtcor, '')
    ts = ts.replace('.h5', '')

    year = ts[:4]
    
    print(f'file from year {year}, {ts}')

    if count_print < 10:
        rdr = load_h5(radar_dir + '/{}/{}'.format(year, f))
        r_i = rain_intensity(rdr)
        print(r_i)
        print(min(r_i[0]))
        print(max(r_i[0]))
        print(len(r_i))
        print(len(r_i[0]))
        print(type(r_i))
        print("---------------")
    count_print = count_print+1

#"""
