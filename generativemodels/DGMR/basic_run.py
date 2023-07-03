from datetime import datetime
import tensorflow as tf
import numpy as np
import wandb
from wandb.keras import WandbCallback
from model_builder_DGMR import DGMR
from batchcreator_DGMR import DataGenerator, get_list_IDs
import config_DGMR
import sys
import logger

# Add parent directory to system path in order to import custom modules
sys.path.append('/')
sys.path.append('/')

print('Starting basic run')
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

# Set hyperparameters
N_EPOCHS = 200
x_length = 6
y_length = 6 #1,3,6 
filter_no_rain = 'avg0.01mm' 
architecture = 'DGMR'
g_cycles = 3
#label_smoothing = 0.1
lr_g = 0.00015 
lr_d = 0.00050
extended_balanced_loss = True
balanced_loss = False
rmse_loss = False
hinge_loss = False
temp_data = True
batch_size = 16

# Loads preproccesed files:
load_prep = True
# Set rtcor as target (instead of aart)
y_is_rtcor= True

# Select filename between a start date and end data
list_IDs_all = np.load(config_DGMR.dir_basic_IDs, allow_pickle=  True)
print(f"Number of radar images (heavy rain): {len(list_IDs_all)}")
amount_samples = 200
list_IDs = list_IDs_all[:amount_samples] # reduce dataset size for testing purposes
print(f"Number of training radar images (heavy rain): {len(list_IDs)}")

#---------- Run with wandb ----------------
run = wandb.init(project='high-precipitation-forecasting',
            config={
            'batch_size' : batch_size,
            'epochs': N_EPOCHS,
            'lr_g': lr_g,
            'lr_d': lr_d,
            'l_adv': l_adv,
            'l_rec': l_rec,
            'g_cycles': g_cycles,
            'label_smoothing': label_smoothing,
            'x_length': x_length,
            'y_length': y_length,
            'rnn_type': 'GRU',
            'filter_no_rain': filter_no_rain,
            'train_data': config_DGMR.dir_basic_IDs[:amount_samples],
            'val_data': config_DGMR.dir_basic_IDs[:amount_samples],
            'architecture': architecture,
            'model': 'DGMR',
            'norm_method': 'minmax',
            'downscale256': True,
            'convert_to_dbz': True,
            'load_prep': load_prep,
            'server':  'RU',
            'rec_with_mae': False,
            'y_is_rtcor': y_is_rtcor,
            'balanced_loss': balanced_loss,
            'extended_balanced_loss': extended_balanced_loss,
            'hinge_loss': hinge_loss,
            'rmse_loss': rmse_loss,
            'temp_data': temp_data,
        })
config = wandb.config
model_path = 'saved_models/model_{}'.format(wandb.run.name.replace('-','_'))

generator = DataGenerator(list_IDs, batch_size=config.batch_size,
                          x_seq_size=config.x_length, y_seq_size=config.y_length,
                          norm_method = config.norm_method, load_prep=config.load_prep,
                          downscale256 = config.downscale256, convert_to_dbz = config.convert_to_dbz, y_is_rtcor = config.y_is_rtcor, temp_data = False)
    
model = DGMR(rnn_type = config.rnn_type, x_length = config.x_length, y_length = config.y_length,
             architecture = config.architecture, g_cycles=config.g_cycles, label_smoothing = config.label_smoothing,
                l_adv = config.l_adv, l_rec = config.l_rec, norm_method = config.norm_method, downscale256 = config.downscale256,
               rec_with_mae = config.rec_with_mae, 
               balanced_loss = balanced_loss, extended_balanced_loss = extended_balanced_loss, rmse_loss = rmse_loss, hinge_loss = hinge_loss, 
               temp_data = temp_data)
model.compile(lr_g = config.lr_g, lr_d = config.lr_d)

 
print("Fit model")
history = model.fit(generator, epochs = config.epochs, callbacks=[WandbCallback()])
