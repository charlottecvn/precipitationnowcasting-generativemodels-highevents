from pyidw import idw
from sklearn.metrics import mean_squared_error
import rasterio
from matplotlib import pyplot as plt

'''
Temperature dtaa should have format
        StationID
        LAT
        LON
        temperature
'''

n_temp_files = 700000
point_file = pd.read_csv("KNMI_temperature_data")

output_idw_temp = []

for i in range(700000):
    point_file_i = point_file[i]

    idw_output = idw.idw_interpolation(
        input_point_file=point_file_i,
        extent_shapefile="example.shp",
        column_name="temperature",
        power=2,
        search_radious=20,
        output_resolution=765,
    )
    
    output_idw_temp[i] = idw_output
    
    
        
