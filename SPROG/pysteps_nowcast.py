from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import conversion, transformation
from pysteps import nowcasts
import pysteps
import config_SPROG
import h5py

def get_mask_rtcor():
    '''
    This function returns the mask of the rtcor data.
    Pixels outside of the radar range have value of 0 and 
    pixels inside radar range have value of 1.
    Mask an prediction by multiplying it with this mask:
    y_pred = y_pred * mask
    '''
    path = config.dir_rtcor +  '2019/01/{}201901010000.h5'.format(config.prefix_rtcor)
    with h5py.File(path, 'r') as f:
        rain = f['image1']['image_data'][:]
        mask = ~(rain == 65535)
    return mask

def sprog_forecast(R, metadata, mask):
    '''
    Performans nowcasting by using the S-PROG algorithm. Predictict 30, 60 and 90m ahead
    R: input sequence of rain data
    '''    
    # Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h (smallest non-zero value is .12),
    # set the fill value to -15 dBR
    R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)
    V = dense_lucaskanade(R)

    # The S-PROG nowcast
    nowcast_method = nowcasts.get_method("sprog")
    
    # forcast for the 30m, 60m and 90m leadtime
    timesteps = [5,11,17]

    R_f = nowcast_method(
          R,
          V,
          timesteps=timesteps,
          n_cascade_levels=8,
          R_thr=-10.0,
          decomp_method="fft",
          bandpass_filter_method="gaussian",
          probmatching_method="mean",
          )
    # Back-transform to rain rate
    R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]
    
    # mask the prediction:
    R_f = R_f * mask
    return R_f
