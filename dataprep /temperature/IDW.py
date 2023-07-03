from pyidw import idw
from sklearn.metrics import mean_squared_error
import rasterio
from matplotlib import pyplot as plt
import config

'''
Temperature dtaa should have format
        StationID
        LAT
        LON
        temperature
'''

n_temp_files = config.n_temp_files
point_file = pd.read_csv("KNMI_temperature_data")

output_idw_temp = []

for i in range(n_temp_files):
    point_file_i = point_file[i*30:(i+1)*30]

    idw_output = idw.idw_interpolation(
        input_point_file=point_file_i,
        extent_shapefile="example.shp",
        column_name="temperature",
        power=2,
        search_radious=5,
        output_resolution=765,
    )
    
    output_idw_temp[i] = idw_output
    
    
        
