# Variables to set for the specific user
USER_FOLDER = 'charlottecvn'

# When not rendering a new list with IDs, use the default option as listed below
basic_IDs_npy = 'list_IDs201520_avg001mm.npy' #"listIDs_prepped_20152020.npy"

# Global variables that point to the correct directory
path_data = '/Volumes/lacie_msc_datascience/msc_thesis/ceph_knmimo/'
path_code = f'/Users/{USER_FOLDER}/Programming/PyCharm/PycharmProjects/MSc thesis/precipitation-nowcasting-GANs-RU/'

path_project = '//'

dir_basic_IDs = path_code + f'/data/{basic_IDs_npy}'

dir_rtcor = path_data + 'dataset_rtcor/'

dir_prep = 'preprocessed/'
dir_rtcor_prep = path_data + dir_prep + 'rtcor_prep/' #rtcor_prep

dir_labels = path_data + dir_prep + 'rtcor_rain_labels/' #prep
dir_labels_heavy = path_data + dir_prep + 'rtcor_heavy_rain_labels/' #prep

prefix_rtcor = 'RAD_NL25_RAC_5M_'
