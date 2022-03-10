from preprocessing.preprocessing_rtcor import *
import config_GAN

# Location where all IDs of the 'rainy' images are specified (complete so not split is applied)
location_list_IDs = '/Users/charlottecvn/Programming/PyCharm/PycharmProjects/MSc thesis/PrecipitationNowcasting-HighEvents-MScThesis/data/list_IDs201520_avg001mm.npy'

# List the names of all the images
fn_rtcor_train = np.load(location_list_IDs, allow_pickle = True)[:,0]
fn_rtcor_val = np.load(location_list_IDs, allow_pickle = True)[:,1]

filenames_rtcor= np.append(fn_rtcor_train, fn_rtcor_val)
filenames_rtcor = [item for sublist in filenames_rtcor for item in sublist]

# Run preprocessing steps (files will be saved in the specified folder in the config
rtcor2npy(config_GAN.dir_rtcor, config_GAN.dir_rtcor_prep, overwrite = False, preprocess = True, filenames = filenames_rtcor)