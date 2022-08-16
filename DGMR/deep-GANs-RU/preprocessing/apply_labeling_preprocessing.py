from preprocessing.preprocessing_rtcor import *
import config_DGMR
from tqdm import tqdm

# Location where all IDs of the 'rainy' images are specified (complete so not split is applied)
location_list_IDs = config_DGMR.dir_basic_IDs

# List the names of all the images
fn_rtcor_train = np.load(location_list_IDs, allow_pickle = True)[:,0]
fn_rtcor_val = np.load(location_list_IDs, allow_pickle = True)[:,1]

filenames_rtcor= np.append(fn_rtcor_train, fn_rtcor_val)
filenames_rtcor = [item for sublist in filenames_rtcor for item in sublist]

#print((filenames_rtcor[0]))
print(f"Total number of listedID radar images: {len(filenames_rtcor)}")

# Run preprocessing steps (files will be saved in the specified folder in the config
#rtcor2npy(config_DGMR.dir_rtcor, config_DGMR.dir_rtcor_prep, overwrite = False, preprocess = True, filenames = filenames_rtcor)

#----------
from os import listdir
from os.path import isfile, join

#print(config_DGMR.dir_rtcor_prep)
files_prep = sorted([f[:12] for f in listdir(config_DGMR.dir_rtcor_prep) if isfile(join(config_DGMR.dir_rtcor_prep, f))])
#print(files_prep[0])
print(f"Total number of prepped radar images: {len(files_prep)}")

print("Difference")
print(len(filenames_rtcor)-len(files_prep))
print(len(np.setdiff1d(np.array(filenames_rtcor), np.array(files_prep))))
#print(len(np.setdiff1d(np.array(files_prep), np.array(filenames_rtcor))))
print((np.setdiff1d(np.array(filenames_rtcor), np.array(files_prep))))

np.save('../data/listIDs_prepped_20152020.npy', files_prep)
