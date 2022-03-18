from datetime import datetime
import tensorflow as tf
import numpy as np

# Add parent directory to system path in order to import custom modules
import sys
sys.path.append('/')
sys.path.append('/')

from model_builder_GAN import GAN
from batchcreator_GAN import DataGenerator, get_list_IDs
import config_GAN

# Set hyperparameters
x_length = 6
y_length = 3
filter_no_rain = 'sum30mm'#'avg0.01mm'
architecture = 'AENN'
l_adv = 0.003
l_rec = 1
g_cycles = 3
label_smoothing = 0.2
lr_g = 0.003
lr_d = 0.001

# Loads preproccesed files:
load_prep = True
# Set rtcor as target (instead of aart)
y_is_rtcor= True

# Select filename between a start date and end data
list_IDs = np.load(config_GAN.dir_basic_IDs, allow_pickle=  True)
list_IDs = list_IDs[:100] # reduce dataset size for testing purposes

model = GAN(rnn_type='GRU', x_length=x_length,
            y_length=y_length, architecture=architecture, relu_alpha=.2,
           l_adv = l_adv, l_rec = l_rec, g_cycles=3, label_smoothing=label_smoothing,
           norm_method = 'minmax', downscale256 = True, rec_with_mae= False,
           batch_norm = False)
print("Compile model")
model.compile(lr_g = lr_g, lr_d = lr_d)

print("Create data generator")
generator = DataGenerator(list_IDs, batch_size=8, x_seq_size=x_length,
                                       y_seq_size=y_length, load_prep=load_prep, y_is_rtcor= y_is_rtcor)
print("Fit model")
hist = model.fit(generator, epochs=1)
