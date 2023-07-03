import tensorflow as tf
import numpy as np
import netCDF4
import os
import config_DGMR as conf
import sys
sys.path.insert(0,conf.path_project)
sys.path.insert(0, '../..')
from ConvGRU2D import ConvGRU2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint
from RepeatVector4D import RepeatVector4D

import sonnet as snt

from scipy.spatial.distance import cityblock


# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    # TODO: update 
    return backend.mean(y_true * y_pred)

# implementation discriminator hinge loss
def loss_hinge_disc(score_generated, score_real):
  # TODO: update 
  return loss
  
# implementation generator hinge loss
def loss_hinge_gen(score_generated):
  # TODO: update 
  loss = -tf.reduce_mean(score_generated)
  return loss
 
# grid cell regularizer using code from deepmind
def grid_cell_regularizer(generated_samples, batch_targets):
  loss = cityblock(generated_samples, batch_targets) 
  return loss

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)
    
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

    
def get_mask_y():
    #The Overeem images are masked. Only values near the netherlands are kept.
    #The model output should also be masked, such that the output of the masked values becomes zero.

    path_mask = 'mask.npy'

    if os.path.isfile(path_mask):
        mask = np.load(path_mask)
    else:
        # Get the mask for the input data
        y_path = conf.dir_aartS
        # The mask is the same for all radar scans, so simply chose a random one to get the mask
        path = y_path + '2019/' + conf.prefix_aart + '201901010000.nc'

        with netCDF4.Dataset(path, 'r') as f:
            rain = f['image1_image_data'][:].data
            mask = (rain != 65535)
        mask = mask.astype(float)
        mask = np.expand_dims(mask, axis=-1)
        mask = crop_center(mask)
        np.save(path_mask,mask)
    return mask

def crop_center(img,cropx=350,cropy=384):
    # batch size, sequence, height, width, channels
     # Only change height and width
    _, y,x, _ = img.shape
    startx = 20+x//2-(cropx//2)
    starty = 40+y//2-(cropy//2)
    return img[:,starty:starty+cropy,startx:startx+cropx:,]

# 2D convolution with spectral normalisation  using code from deepmind
class SNConv2D:
  def __init__(self, output_channels, kernel_size, stride=1, rate=1,
               padding='SAME', sn_eps=0.0001, use_bias=True):
    # constructer
    self._output_channels = output_channels
    self._kernel_size = kernel_size
    self._stride = stride
    self._rate = rate
    self._padding = padding
    self._sn_eps = sn_eps
    self._initializer = tf.orthogonal_initializer
    self._use_bias = use_bias

  def __call__(self, tensor):
    SNConv2D = snt.wrap_with_spectral_norm(snt.Conv2D, {'eps': 1e-4})
    return SNConv2D
    
# latent conditioning stack for sampler using code from deepmind
class LatentCondStack(object):

  def __init__(self):
    self._conv1 = SNConv2D(output_channels=8, kernel_size=3)
    self._lblock1 = LBlock(output_channels=24)
    self._lblock2 = LBlock(output_channels=48)
    self._lblock3 = LBlock(output_channels=192)
    self._mini_attn_block = Attention(num_channels=192)
    self._lblock4 = LBlock(output_channels=768)

  def __call__(self, batch_size, resolution=(256, 256)):
    # Independent draws from a Normal distribution.
    h, w = resolution[0] // 32, resolution[1] // 32
    z = tf.random.normal([batch_size, h, w, 8])

    # 3x3 convolution.
    z = self._conv1(z)

    # Three L Blocks to increase the number of channels to 24, 48, 192.
    z = self._lblock1(z)
    z = self._lblock2(z)
    z = self._lblock3(z)

    # Spatial attention module.
    z = self._mini_atten_block(z)

    # L Block to increase the number of channels to 765.
    z = self._lblock4(z)

    return z

# residual block for the Latent Stack using code from deepmind
class LBlock(object):
  def __init__(self, output_channels, kernel_size=3, conv=tf.keras.layers.Conv2D,
               activation=tf.nn.relu):
    # Constructor for the D blocks of the DVD-GAN.
    self._output_channels = output_channels
    self._kernel_size = kernel_size
    self._conv = conv
    self._activation = activation

  def __call__(self, inputs):
    # Stack of two conv. layers and nonlinearities that increase the number of
    # channels.
    h0 = self._activation(inputs)
    h1 = self._conv(num_channels=self.output_channels,
                    kernel_size=self._kernel_size)(h0)
    h1 = self._activation(h1)
    h2 = self._conv(num_channels=self._output_channels,
                    kernel_size=self._kernel_size)(h1)

    # Prepare the residual connection branch.
    input_channels = h0.shape.as_list()[-1]
    if input_channels < self._output_channels:
      sc = self._conv(num_channels=self._output_channels - input_channels,
                      kernel_size=1)(inputs)
      sc = tf.concat([inputs, sc], axis=-1)
    else:
      sc = inputs

    # Residual connection.
    return h2 + sc
    
# attention summation using code from deepmind
def attention_einsum(q, k, v):
  # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
  k = tf.reshape(k, [-1, k.shape[-1]])  # [h, w, c] -> [L, c]
  v = tf.reshape(v, [-1, v.shape[-1]])  # [h, w, c] -> [L, c]

  # Einstein summation corresponding to the query * key operation.
  beta = tf.nn.softmax(tf.einsum('hwc, Lc->hwL', q, k), axis=-1)

  # Einstein summation corresponding to the attention * value operation.
  out = tf.einsum('hwL, Lc->hwc', beta, v)
  return out

# layer for applying an operation on each element, along a specified axis; using code from deepmind
class ApplyAlongAxis:
  def __init__(self, operation, axis=0):
    # constructor
    self._operation = operation
    self._axis = axis

  def __call__(self, *args):
    split_inputs = [tf.unstack(arg, axis=self._axis) for arg in args]
    res = [self._operation(x) for x in zip(*split_inputs)]
    return tf.stack(res, axis=self._axis)

# attention module using code from deepmind
class Attention(object):
  def __init__(self, num_channels, ratio_kq=8, ratio_v=8, conv=tf.keras.layers.Conv2D):
    # constructor
    self._num_channels = num_channels
    self._ratio_kq = ratio_kq
    self._ratio_v = ratio_v
    self._conv = conv

    # Learnable gain parameter
    self._gamma = tf.get_variable(
        'miniattn_gamma', shape=[],
        initializer=tf.initializers.zeros(tf.float32))

  def __call__(self, tensor):
    # Compute query, key and value using 1x1 convolutions.
    query = self._conv(
        output_channels=self._num_channels // self._ratio_kq,
        kernel_size=1, padding='VALID', use_bias=False)(tensor)
    key = self._conv(
        output_channels=self._num_channels // self._ratio_kq,
        kernel_size=1, padding='VALID', use_bias=False)(tensor)
    value = self._conv(
        output_channels=self._num_channels // self._ratio_v,
        kernel_size=1, padding='VALID', use_bias=False)(tensor)

    # Apply the attention operation.
    out = ApplyAlongAxis(attention_einsum, axis=0)(query, key, value)
    out = self._gamma * self._conv(
        output_channels=self._num_channels,
        kernel_size=1, padding='VALID', use_bias=False)(out)

    # Residual connection.
    return out + tensor

# residual generator block using code from deepmind
class GBlock(object):
  def __init__(self, output_channels, sn_eps=0.0001):
    self._conv1_3x3 = SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn1 = tf.keras.layers.BatchNormalization()
    self._conv2_3x3 = SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn2 = tf.keras.layers.BatchNormalization()
    self._output_channels = output_channels
    self._sn_eps = sn_eps

  def __call__(self, inputs):
    input_channels = inputs.shape[-1]

    # Optional spectrally normalized 1x1 convolution.
    if input_channels != self._output_channels:
      conv_1x1 = SNConv2D(
          self._output_channels, kernel_size=1, sn_eps=self._sn_eps)
      sc = conv_1x1(inputs)
    else:
      sc = inputs

    # Two-layer residual connection, with batch normalization, nonlinearity and
    # 3x3 spectrally normalized convolution in each layer.
    h = tf.nn.relu(self._bn1(inputs))
    h = self._conv1_3x3(h)
    h = tf.nn.relu(self._bn2(h))
    h = self._conv2_3x3(h)

    # Residual connection.
    return h + sc

# upsampling residual generator block using code from deepmind
class UpsampleGBlock(object):
  def __init__(self, output_channels, sn_eps=0.0001):
    self._conv_1x1 = SNConv2D(
        output_channels, kernel_size=1, sn_eps=sn_eps)
    self._conv1_3x3 = SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn1 = tf.keras.layers.BatchNormalization()
    self._conv2_3x3 = SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn2 = tf.keras.layers.BatchNormalization()
    self._output_channels = output_channels

  def __call__(self, inputs):
    # x2 upsampling and spectrally normalized 1x1 convolution.
    sc = tf.keras.layers.UpSampling2D(inputs, interpolation='nearest')
    sc = self._conv_1x1(sc)

    # Two-layer residual connection, with batch normalization, nonlinearity and
    # 3x3 spectrally normalized convolution in each layer, and x2 upsampling in
    # the first layer.
    h = tf.nn.relu(self._bn1(inputs))
    h = tf.keras.layers.UpSampling2D(h, interpolation='nearest')
    h = self._conv1_3x3(h)
    h = tf.nn.relu(self._bn2(h))
    h = self._conv2_3x3(h)

    # Residual connection.
    return h + sc

# conditioning stack generator block using code from deepmind
class ConditioningStack(object):

  def __init__(self):
    self._block1 = CBlock(output_channels=48, downsample=True)
    self._conv_mix1 = SNConv2D(output_channels=48, kernel_size=3)
    self._block2 = CBlock(output_channels=96, downsample=True)
    self._conv_mix2 = SNConv2D(output_channels=96, kernel_size=3)
    self._block3 = CBlock(output_channels=192, downsample=True)
    self._conv_mix3 = SNConv2D(output_channels=192, kernel_size=3)
    self._block4 = CBlock(output_channels=384, downsample=True)
    self._conv_mix4 = SNConv2D(output_channels=384, kernel_size=3)

  def __call__(self, inputs):
    # Space to depth conversion of 256x256x1 radar to 128x128x4 hiddens.
    h0 = batch_apply(
        functools.partial(tf.nn.space_to_depth, block_size=2), inputs)

    # Downsampling residual D Blocks.
    h1 = ApplyAlongAxis(self._block1, h0)
    h2 = ApplyAlongAxis(self._block2, h1)
    h3 = ApplyAlongAxis(self._block3, h2)
    h4 = ApplyAlongAxis(self._block4, h3)

    # Spectrally normalized convolutions, followed by rectified linear units.
    init_state_1 = self._mixing_layer(h1, self._conv_mix1)
    init_state_2 = self._mixing_layer(h2, self._conv_mix2)
    init_state_3 = self._mixing_layer(h3, self._conv_mix3)
    init_state_4 = self._mixing_layer(h4, self._conv_mix4)

    # Return a stack of conditioning representations of size 64x64x48, 32x32x96,
    # 16x16x192 and 8x8x384.
    return init_state_1, init_state_2, init_state_3, init_state_4

  def _mixing_layer(self, inputs, conv_block):
    # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
    # then perform convolution on the output while preserving number of c.
    stacked_inputs = tf.concat(tf.unstack(inputs, axis=1), axis=-1)
    return tf.nn.relu(conv_block(stacked_inputs))

# sampler generator block using code from deepmind
class Sampler(object):
  def __init__(self, lead_time=90, time_delta=5):
    self._num_predictions = lead_time // time_delta
    self._latent_stack = LatentCondStack()

    self._conv_gru4 = ConvGRU2D.ConvGRU2D()
    self._conv4 = SNConv2D(kernel_size=1, output_channels=768)
    self._gblock4 = GBlock(output_channels=768)
    self._g_up_block4 = UpsampleGBlock(output_channels=384)

    self._conv_gru3 = ConvGRU2D.ConvGRU2D
    self._conv3 = SNConv2D(kernel_size=1, output_channels=384)
    self._gblock3 = GBlock(output_channels=384)
    self._g_up_block3 = UpsampleGBlock(output_channels=192)

    self._conv_gru2 = ConvGRU2D.ConvGRU2D
    self._conv2 = SNConv2D(kernel_size=1, output_channels=192)
    self._gblock2 = GBlock(output_channels=192)
    self._g_up_block2 = GBlock(output_channels=96)

    self._conv_gru1 = ConvGRU2D.ConvGRU2D
    self._conv1 = SNConv2D(kernel_size=1, output_channels=96)
    self._gblock1 = GBlock(output_channels=96)
    self._g_up_block1 = UpsampleGBlock(output_channels=48)

    self._bn = tf.keras.layers.BatchNormalization()
    self._output_conv = SNConv2D(kernel_size=1, output_channels=4)

  def __call__(self, initial_states, resolution):
    init_state_1, init_state_2, init_state_3, init_state_4 = initial_states
    batch_size = init_state_1.shape.as_list()[0]

    # Latent conditioning stack.
    z = self._latent_stack(batch_size, resolution)
    hs = [z] * self._num_predictions

    # Layer 4 (bottom-most).
    hs, _ = tf.nn.static_rnn(self._conv_gru4, hs, init_state_4)
    hs = [self._conv4(h) for h in hs]
    hs = [self._gblock4(h) for h in hs]
    hs = [self._g_up_block4(h) for h in hs]

    # Layer 3.
    hs, _ = tf.nn.static_rnn(self._conv_gru3, hs, init_state_3)
    hs = [self._conv3(h) for h in hs]
    hs = [self._gblock3(h) for h in hs]
    hs = [self._g_up_block3(h) for h in hs]

    # Layer 2.
    hs, _ = tf.nn.static_rnn(self._conv_gru2, hs, init_state_2)
    hs = [self._conv2(h) for h in hs]
    hs = [self._gblock2(h) for h in hs]
    hs = [self._g_up_block2(h) for h in hs]

    # Layer 1 (top-most).
    hs, _ = tf.nn.static_rnn(self._conv_gru1, hs, init_state_1)
    hs = [self._conv1(h) for h in hs]
    hs = [self._gblock1(h) for h in hs]
    hs = [self._g_up_block1(h) for h in hs]

    # Output layer.
    hs = [tf.nn.relu(self._bn(h)) for h in hs]
    hs = [self._output_conv(h) for h in hs]
    hs = [tf.nn.depth_to_space(h, 2) for h in hs]

    return tf.stack(hs, axis=1)
 
 # convolutional residual block using code from deepmind
 class CBlock(object):
  def __init__(self, output_channels, kernel_size=3, downsample=True,
               pre_activation=True, conv=SNConv2D,
               pooling=tf.keras.layers.AveragePooling2D, activation=tf.nn.relu):
    # Constructor for the D blocks of the DVD-GAN.
    self._output_channels = output_channels
    self._kernel_size = kernel_size
    self._downsample = downsample
    self._pre_activation = pre_activation
    self._conv = conv
    self._pooling = pooling
    self._activation = activation

  def __call__(self, inputs):
    h0 = inputs

    # Pre-activation.
    if self._pre_activation:
      h0 = self._activation(h0)

    # First convolution.
    input_channels = h0.shape.as_list()[-1]
    h1 = self._conv(num_channels=input_channels,
                    kernel_size=self._kernel_size)(h0)
    h1 = self._activation(h1)

    # Second convolution.
    h2 = self._conv(num_channels=self._output_channels,
                    kernel_size=self._kernel_size)(h1)

    # Downsampling.
    if self._downsample:
      h2 = self.pooling(h2)

    # The residual connection, make sure it has the same dimensionality
    # with additional 1x1 convolution and downsampling if needed.
    if input_channels != self._output_channels or self._downsample:
      sc = self._conv(num_channels=self._output_channels,
                      kernel_size=1)(inputs)
      if self.downsample:
        sc = self.pooling(sc)
    else:
      sc = inputs

    # Residual connection.
    return h2 + sc

# Spatial Discriminator block using code from deepmind
class SpatialDiscriminator(object):
  def __init__(self):
    pass

  def __call__(self, frames):
    b, n, h, w, c = tf.shape(frames).as_list()

    # Process each of the n inputs independently.
    frames = tf.reshape(frames, [b * n, h, w, c])

    # Space-to-depth stacking from 128x128x1 to 64x64x4.
    frames = tf.nn.space_to_depth(frames, block_size=2)

    # Five residual D Blocks to halve the resolution of the image and double
    # the number of channels.
    y = CBlock(output_channels=48, pre_activation=False)(frames)
    y = CBlock(output_channels=96)(y)
    y = CBlock(output_channels=192)(y)
    y = CBlock(output_channels=384)(y)
    y = CBlock(output_channels=768)(y)

    # One more D Block without downsampling or increase in number of channels.
    y = CBlock(output_channels=768, downsample=False)(y)

    # Sum-pool the representations and feed to spectrally normalized lin. layer.
    y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
    y = tf.keras.layers.BatchNormalization()(y)
    output_layer = tf.keras.layers.Dense(1)
    output = output_layer(y)

    # Take the sum across the t samples. Note: we apply the ReLU to
    # (1 - score_real) and (1 + score_generated) in the loss.
    output = tf.reshape(output, [b, n, 1])
    output = tf.reduce_sum(output, keepdims=True, axis=1)
    return output

# Temporal Discriminator block using code from deepmind
class TemporalDiscriminator(object):
  def __init__(self):
    pass

  def __call__(self, frames):
    b, ts, hs, ws, cs = tf.shape(frames).as_list()

    # Process each of the ti inputs independently.
    frames = tf.reshape(frames, [b * ts, hs, ws, cs])

    # Space-to-depth stacking from 128x128x1 to 64x64x4.
    frames = tf.nn.space_to_depth(frames, block_size=2)

    # Stack back to sequences of length ti.
    frames = tf.reshape(frames, [b, ts, hs, ws, cs])

    # Two residual 3D Blocks to halve the resolution of the image, double
    # the number of channels, and reduce the number of time steps.
    y = CBlock(output_channels=48, conv=SNConv3D,
               pooling=tf.keras.layers.AveragePooling3D,
               pre_activation=False)(frames)
    y = CBlock(output_channels=96, conv=SNConv3D,
               pooling=tf.keras.layers.AveragePooling3D)(y)

    # Get t < ts, h, w, and c, as we have downsampled in 3D.
    _, t, h, w, c = tf.shape(frames).as_list()

    # Process each of the t images independently.
    # b t h w c -> (b x t) h w c
    y = tf.reshape(y, [-1] + [h, w, c])

    # Three residual D Blocks to halve the resolution of the image and double
    # the number of channels.
    y = CBlock(output_channels=192)(y)
    y = CBlock(output_channels=384)(y)
    y = CBlock(output_channels=768)(y)

    # One more D Block without downsampling or increase in number of channels.
    y = CBlock(output_channels=768, downsample=False)(y)

    # Sum-pool the representations and feed to spectrally normalized lin. layer.
    y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
    y = tf.keras.layers.BatchNormalization(calc_sigma=False)(y)
    output_layer = tf.keras.layers.Dense(1)
    output = output_layer(y)

    # Take the sum across the t samples. Note: we apply the ReLU to
    # (1 - score_real) and (1 + score_generated) in the loss.
    output = tf.reshape(output, [b, t, 1])
    scores = tf.reduce_sum(output, keepdims=True, axis=1)
    return scores


def generator_DGMR(x, rnn_type='GRU', relu_alpha=0.2, x_length=6, y_length=1, norm_method = None, downscale256 = False, batch_norm = False, num_filters = 32):
    '''
    This generator uses similar architecture as in DGMR.
    '''
    lead_time=90
    time_delta=5
    
    # Add padding to make square image
    x = tf.keras.layers.ZeroPadding3D(padding=(0,0,34))(x)
    # apply stacks
    init_states = ConditioningStack(x)
    # create output using sampler
    output = Sampler(lead_time, time_delta, initial_states)
    
    output = tf.keras.layers.Cropping2D((0,17))(output)

    return output

def discriminator_DGMR(x, relu_alpha,  wgan = False, downscale256 = False, batch_norm = False, drop_out = False):
    x = tf.keras.layers.ZeroPadding3D(padding=(0,0,17))(x)
    
    # spatial discriminator
    b, t, h, w, c = tf.shape(x).as_list()
    target_frames_sel = tf.range(6, t)
    permutation = tf.stack([
        tf.random_shuffle(target_frames_sel)[:8]
        for _ in range(b)
    ], 0)
    frames_for_sd = tf.gather(x, permutation, batch_dims=1)
    frames_for_sd = tf.layers.average_pooling3d(
        frames_for_sd, [1, 2, 2], [1, 2, 2], data_format='channels_last')
    sd_out = SpatialDiscriminator(frames_for_sd)

    # temporal discriminator
    h_offset = tf.random_uniform([], 0, (2 - 1) * (h // 2), tf.int32)
    w_offset = tf.random_uniform([], 0, (2 - 1) * (w // 2), tf.int32)
    zero_offset = tf.zeros_like(w_offset)
    begin_tensor = tf.stack(
        [zero_offset, zero_offset, h_offset, w_offset, zero_offset], -1)
    size_tensor = tf.constant([b, t, h // 2, w // 2, c])
    frames_for_td = tf.slice(x, begin_tensor, size_tensor)
    frames_for_td.set_shape([b, t, h // 2, w // 2, c])
    td_out = self._temporal_discriminator(frames_for_td)

    output = tf.concat([sd_out, td_out], 1)
    
    return output

def build_generator(rnn_type, relu_alpha, x_length=6, y_length=1, architecture='DGMR',
                    norm_method = None, downscale256 = False, batch_norm = False, num_filters = 32):
    inp_dim = (768, 700,1)
    out_dim = (384, 350, 1)
    if downscale256:
        inp_dim = (256, 256, 1)
    inp = tf.keras.Input(shape=(x_length, *inp_dim))

    output = generator_DGMR(inp, rnn_type, relu_alpha,
                                x_length, y_length, norm_method=norm_method,
                                downscale256 = downscale256, batch_norm = batch_norm, num_filters = num_filters)
    
    if not downscale256:  # Mask pixels outside Netherlands
        mask = tf.constant(get_mask_y(), 'float32')
        output = tf.keras.layers.Lambda(lambda x: x * mask, name='Mask')(output)
        if norm_method and norm_method == 'minmax_tanh':
            output = tf.keras.layers.subtract([output, 1-mask])
    
    model = tf.keras.Model(inputs=inp, outputs=output, name='Generator')
    return model

def build_discriminator(relu_alpha, y_length, architecture = 'DGMR', wgan = False, downscale256 = False, batch_norm = False,
                       drop_out = False):
    inp_dim = (384, 350, 1)
    if downscale256:
        inp_dim = (256, 256, 1)
    inp = tf.keras.Input(shape=(y_length, *inp_dim))
    
    output = discriminator_DGMR(inp, relu_alpha, wgan, downscale256 = downscale256,
                                    batch_norm = batch_norm, drop_out = drop_out)

    model = tf.keras.Model(inputs=inp, outputs=output, name='Discriminator')
    return model

class DGMR(tf.keras.Model):
    def __init__(self, inp_dim = (768,700,1), out_dim = (384, 350, 1), rnn_type='GRU', x_length=6,
                 y_length=1, relu_alpha=0.2, architecture='DGMR', l_adv = 1, l_rec = 0.01, g_cycles=1,
                 label_smoothing = 0, norm_method = None, wgan = False, downscale256 = False, rec_with_mae=True,
                 batch_norm = False, drop_out = False, r_to_dbz = False, balanced_loss = False,
                 rmse_loss = False, temp_data = False, SPROG_data = False):
        '''
        inp_dim: dimensions of input image(s), default 768x700
        out_dim: dimensions of the output image(s), default 384x350
        rnn_type: type of recurrent neural network can be LSTM or GRU
        x_length: length of input sequence
        y_length: length of output sequence
        relu_alpha: slope of leaky relu layers
        architecture: either 'DGMR' or ''
        l_adv: weight of the adverserial loss for generator
        l_rec: weight of reconstruction loss (mse + mae) for the generator
        g_cycles: how many cycles to train the generator per train cycle
        label_smoothing: When > 0, we compute the loss between the predicted labels
                          and a smoothed version of the true labels, where the smoothing
                          squeezes the labels towards 0.5. Larger values of
                          label_smoothing correspond to heavier smoothing
        norm_method: which normalization method was used.
                     Can be none or minmax_tanh where data scaled to be between -1 and 1
        wgan: Option to use wasserstein loss (Not fully implemented yet)
        downscale256: if true than the images are downscaled to 256x256 by using bilinear interpolation
        rec_with_mae: if true the reconstruction loss is MSE+MAE if false, rec it consists of only the MSE
        batch_norm: if true batch normalization is applied after each convolution(/rnn) block
        drop_out: if true adds dropout layer after each conv block in the Discriminator (dropout rate of 0.2)
        r_to_dbz: If true the data values are in dbz not in r (mm/h)
        balanced_loss: If true, balanced loss will be applied (generator)
        rmse_loss: If true, RMSE loss will be applied (generator)
        temp_data: If true, temperature data will be used as additional feature
        '''
        super(DGMR, self).__init__()

        self.generator = build_generator(rnn_type, x_length=x_length,
                                         y_length = y_length, relu_alpha=relu_alpha,
                                         architecture=architecture, norm_method=norm_method,
                                        downscale256 = downscale256, batch_norm = batch_norm)
        
        self.discriminator_frame = build_discriminator(y_length=1,
                                                 relu_alpha=relu_alpha,
                                                architecture=architecture, wgan = wgan,
                                                 downscale256 = downscale256, batch_norm = batch_norm, drop_out = drop_out)
        self.y_length = y_length
        if y_length > 1:
            self.discriminator_seq = build_discriminator(y_length=x_length+y_length,
                                                     relu_alpha=relu_alpha,
                                                    architecture=architecture, wgan = wgan,
                                                     downscale256 = downscale256, batch_norm = batch_norm, drop_out = drop_out)
        self.l_adv = l_adv
        self.l_rec = l_rec
        self.g_cycles=g_cycles
        self.label_smoothing=label_smoothing
        self.norm_method=norm_method
        self.r_to_dbz = r_to_dbz
        self.wgan = wgan
        self.rec_with_mae = rec_with_mae
        self.downscale256 = downscale256
        self.balanced_loss = balanced_loss
        self.rmse_loss = rmse_loss
        self.temp_data = temp_data
        self.SPROG_data = SPROG_data
        
    def compile(self, lr_g=0.0001, lr_d = 0.0001):
        super(DGMR, self).compile()
        
        self.g_optimizer = Adam(learning_rate=lr_g)
        self.d_optimizer = Adam(learning_rate=lr_d)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.loss_fn_d = tf.keras.losses.BinaryCrossentropy(label_smoothing=self.label_smoothing)
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        self.loss_mae = tf.keras.losses.MeanAbsoluteError()
        
        self.g_loss_metric_frame = tf.keras.metrics.Mean(name="g_loss_frame")
        self.g_loss_metric_seq = tf.keras.metrics.Mean(name="g_loss_seq")
        
        self.d_loss_metric_frame= tf.keras.metrics.Mean(name="d_loss_frame")
        self.d_loss_metric_seq = tf.keras.metrics.Mean(name="d_loss_seq")
        
        self.d_acc_frame = tf.keras.metrics.BinaryAccuracy(name='d_acc_frame')
        self.d_acc_seq = tf.keras.metrics.BinaryAccuracy(name='d_acc_seq')
        
        self.rec_metric = tf.keras.metrics.Mean(name="rec_loss")

        if self.wgan:
            self.opt = RMSprop(lr=0.00005)
            self.loss_fn = wasserstein_loss
            
    def rain_intensity(img):
        '''
        Computes the rain intensity of an image, using to the dBZ and dBR
        The function leads to a numpy array with the intensities
        '''
        b = 1.56 #20
        a = 58.53 #20
        dBZ = img * 70.0 - 10.0
        dBR = (dBZ - 10.0 * np.log10(a)) / b
        return np.power(10, dBR / 10.0)
        
    def tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
            
    def loss_rec(self, target, pred, MAE = True, balanced = False, rmse = False):
        '''
        Reconstruction loss: Hinge loss
        mae: If false the reconstruction loss is equal to the MSE, this was found to perform better
        balanced: If true, balanced loss is applied
        rmse: if true, rmse loss is applied
        '''
        batch_inputs, batch_targets = get_data_batch(batch_size=16)
        grid_cell_reg = grid_cell_regularizer(tf.stack(pred, axis=0))
        gen_disc_loss = loss_hinge_gen(tf.concat(pred, axis=0))
        gen_loss = gen_disc_loss + 20.0 * grid_cell_reg
        
        def tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator
        
        if balanced:
            img = pred
            # Z-R relationship constants
            tf_a = tf.constant(58.53)
            tf_b = tf.constant(1.56*10)
            # convert
            tf_dBZ_0 = tf.math.multiply(img, 70)
            tf_dBZ = tf_dBZ_0 - 10
            tf_dBR_0 = tf.math.multiply(tf.constant(10.), tf_log10(tf_a))
            tf_dBR_1 = tf_dBZ - tf_dBR_0
            tf_dBR = tf.divide(tf_dBR_1, tf_b)
            # weights for the loss
            weights_tf = tf.math.square(tf.pow(10., tf_dBR))
            weights_tf = tf.clip_by_value(weights_tf, 0, 30)
            norm_tf_w = (weights_tf - 30)/(30 - 0)
            norm_tf_w = tf.math.abs(norm_tf_w)
            # compute balanced loss
            tf_diff_mse = tf.math.squared_difference(target, pred, name=None)
            tf_diff_mae = tf.math.abs(target-pred)
            tf_mse_input = tf.math.multiply(norm_tf_w, tf_diff_mse)
            tf_mae_input = tf.math.multiply(norm_tf_w, tf_diff_mae)
            tf_bmse = tf.reduce_mean(tf_mse_input)
            tf_bmae = tf.reduce_mean(tf_mae_input)
            return tf_bmse+tf_bmae
        elif rmse:
            g_loss_mse = self.loss_mse(target, pred)
            g_loss_mae = self.loss_mae(target, pred)
            return g_loss_mse + g_loss_mae
        else:
            g_loss = gen_loss
        return g_loss
    
    def call(self, x):
        """Run the model."""
        y_pred = self.generator(x)
        return y_pred

    @property
    def metrics(self):
        return [self.d_loss_metric_frame, self.d_loss_metric_seq,
                self.g_loss_metric_frame, self.g_loss_metric_seq,
                self.rec_metric, self.d_acc_frame, self.d_acc_seq]
    
    def train_disc_seq(self, inp, labels, train = True ):
        if train:
            with tf.GradientTape() as tape:
                predictions = self.discriminator_seq(inp)
                d_loss_seq = loss_hinge_disc(predictions, labels)
            grads = tape.gradient(d_loss_seq, self.discriminator_seq.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator_seq.trainable_weights)
            )
        else:
            predictions = self.discriminator_seq(inp)
            d_loss_seq = loss_hinge_disc(predictions, labels)
        # Update D accuracy metric
        self.d_acc_seq.update_state(labels, predictions)
        return d_loss_seq
    
    def train_disc_frame(self, inp, labels, train = True ):
        if train:
            with tf.GradientTape() as tape:
                d_loss_frame = 0
                for i in range(self.y_length):
                    frame = inp[:,i:i+1]
                    predictions = self.discriminator_frame(frame)
                    d_loss_frame += loss_hinge_disc(predictions, labels)
            grads = tape.gradient(d_loss_frame, self.discriminator_frame.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator_frame.trainable_weights)
            )
        else:
            d_loss_frame = 0
            for i in range(self.y_length):
                frame = inp[:,i:i+1]
                predictions = self.discriminator_frame(frame)
                d_loss_frame += loss_hinge_disc(predictions, labels)
                
        # Update D accuracy metric
        # Now d_acc_frame is accuracy on the last frame
        self.d_acc_frame.update_state(labels, predictions)
        return d_loss_frame
    
    def train_discriminators(self, xs , ys, batch_size, train = True):
        # Decode them to fake images
        generated_images = self.generator(xs)

        # Combine them with real images
        combined_images = tf.concat([generated_images, ys], axis=0)
        
        # concatenate input and predictions in feature dimensions
        # D then looks at the whole sequence (cGAN)
        seq_pred = tf.concat([xs, generated_images], axis=1)
        seq_real = tf.concat([xs, ys], axis=1)
        combined_sequences = tf.concat([seq_pred, seq_real], axis=0)
        
        # Assemble labels discriminating fake from real images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        
        # Train the frame discriminator
        d_loss_frame = self.train_disc_frame(combined_images, labels, train)
        
        # Train the sequence discriminator
        if self.y_length > 1:
            d_loss_seq = self.train_disc_seq(combined_sequences, labels, train)
        else:
            d_loss_seq = d_loss_frame
        return d_loss_frame, d_loss_seq

    def train_generator(self, xs, ys, batch_size, train = True):
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        if train:
            for _ in range(self.g_cycles):
                with tf.GradientTape() as tape:
                    generated_images = self.generator(xs)
                    adv_loss_frame = self.train_disc_frame(generated_images, misleading_labels, train = False)
                    if self.y_length > 1:
                        seq_pred = tf.concat([xs, generated_images], axis=1)
                        adv_loss_seq = self.train_disc_seq(seq_pred, misleading_labels, train = False)
                    else:
                        adv_loss_seq = adv_loss_frame
                    g_loss_adv = adv_loss_frame + adv_loss_seq
                    g_loss_rec = self.loss_rec(ys, generated_images, self.rec_with_mae, self.balanced_loss, self.rmse_loss)
                    g_loss =  self.l_adv * g_loss_adv  + self.l_rec * g_loss_rec
                grads = tape.gradient(g_loss, self.generator.trainable_weights)
                self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        else:
            generated_images = self.generator(xs)
            adv_loss_frame = self.train_disc_frame(generated_images, misleading_labels, train = False)
            if self.y_length > 1:
                seq_pred = tf.concat([xs, generated_images], axis=1)
                adv_loss_seq = self.train_disc_seq(seq_pred, misleading_labels, train = False)
            else:
                adv_loss_seq = 0
            g_loss_adv = adv_loss_frame + adv_loss_seq
            g_loss_rec = self.loss_rec(ys, generated_images, self.rec_with_mae, self.balanced_loss, self.rmse_loss)
            g_loss =  self.l_adv * g_loss_adv  + self.l_rec * g_loss_rec
        return adv_loss_frame, adv_loss_seq, g_loss_rec
    
    def undo_prep(self, x):
        x = batchcreator.undo_prep(x, norm_method=self.norm_method, r_to_dbz=self.r_to_dbz, downscale256=self.downscale256)
        return x
    
    def model_step(self, batch, train = True):
        '''
        This function performs train_step
        batch: batch of x and y data
        train: wether to train the model
               True for train_step, False when performing test_step
        '''
        xs, ys = batch
        batch_size = tf.shape(xs)[0] #16

        d_loss_frame, d_loss_seq = self.train_discriminators(xs,ys,batch_size,train)
        g_loss_frame, g_loss_seq, g_loss_rec  = self.train_generator(xs,ys,batch_size,train)

        # Update metrics
        self.d_loss_metric_frame.update_state(d_loss_frame)
        self.d_loss_metric_seq.update_state(d_loss_seq)
        self.g_loss_metric_frame.update_state(g_loss_frame)
        self.g_loss_metric_seq.update_state(g_loss_seq)
        self.rec_metric.update_state(g_loss_rec)
         
        if self.y_length > 1:
            return {
                "d_loss_frame": self.d_loss_metric_frame.result(),
                "d_loss_seq": self.d_loss_metric_seq.result(),
                "g_loss_frame": self.g_loss_metric_frame.result(),
                "g_loss_seq": self.g_loss_metric_seq.result(),
                "rec_loss": self.rec_metric.result(),
                "d_acc_frame": self.d_acc_frame.result(),
                'd_acc_seq': self.d_acc_seq.result()
            }
        else:
            return {
            "d_loss_frame": self.d_loss_metric_frame.result(),
            "g_loss_frame": self.g_loss_metric_frame.result(),
            "rec_loss": self.rec_metric.result(),
            "d_acc_frame": self.d_acc_frame.result(),
        }
            
    def train_step(self, batch):
        metric_dict = self.model_step(batch, train = True)
        return metric_dict
    
    def test_step(self, batch):
        metric_dict = self.model_step(batch, train = False)
        return metric_dict
