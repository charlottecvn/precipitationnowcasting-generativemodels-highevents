print("precipitation nowcasting using DGMR and high events")

import tensorflow as tf

print('Starting test run')
physical_devices = tf.config.list_physical_devices('GPU') #GPU
print("Num GPUs Available: ", len(physical_devices))
