# Precipitation nowcasting for high intensity events

Code related to my MSc Thesis at Radboud University for the programme Computing Science. 

Thesis is building and improving upon previous work in precipitation nowcasting:
- ["Skillful Precipitation Nowcasting using Deep Generative Models of Radar"](https://github.com/deepmind/deepmind-research/tree/master/nowcasting) by DeepMind
- ["Precipitation Nowcasting using Generative Adversarial Networks"](https://github.com/KoertS/precipitation-nowcasting-using-GANs) by K. Schreurs et al.

## Data 
Real-time precipitation radar data is used from 2008 till 2020. The data past 2018 is publicly available at the [KNMI data platform](https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m_tar/versions/1.0/files). This dataset was used to train and validate the model

In order to run the code on your machine, you need to specify the following directories from the radboud server (`ceph/knmimo`):
- `./preprocessed/rtcor_prep`
- `./preprocessed/rtcor_rain_labels`
- `./preprocessed/rtcor_heavy_rain_labels`
- `./dataset_rtcor`

In the [`config_DGMR.py`](https://github.com/charlottecvn/sprecipitation-nowcasting-GANs-RU/blob/main/config_GAN.py) change the path to your data (path_data) and to your project (path_project) to match your system. Furthermore the real-time dataset can have different names (rtcor or RAC). Check your data to see if the prefix of your data matches the one stated in the config file (prefix_rtcor).

## Preprocessing 
In order to execute a `basic_run.sh`, preprocessed data is used. When you want to preprocess or label the data yourself, these scripts can found in the `preprocessing` folder.

A subselection of the data was used, only samples with sufficient rain were included. Each sample in the dataset was labeled as rainy or not rainy.
To obtain these labels you can run the python script [`rainyday_labeler.py`](https://github.com/charlottecvn/precipitation-nowcasting-GANs-RU/blob/main/preprocessing/rainyday_labeler.py), with as argument the year you want to label (example: rainyday_labeler.py 2019, this would label all the samples from 2019)

A generator is used to retrieve parts of the data during runtime (Datagenerator class in [batchcreator module](https://github.com/charlottecvn/precipitation-nowcasting-GANs-RU/blob/main/batchcreator_GAN.py)). The generator loads the input, target pairs by filename. To create these input and target pairs you can run the python script [`create_traindata_IDs.py`](https://github./charlottecvn/precipitation-nowcasting-GANs-RU/blob/main/preprocessing/create_traindata_IDs.py) and change the time interval and input and output length and the filename to your needs.
com

## Basic run 
To obtain the results of an basic run, execute the `basic_run.py` file. In order to run the file on the server, use the following steps to login in to the correct server (cn46 or cn47 in the `ceph/knmimo` group), and to activate the slurm jobs 
```
ssh username@lilo8.science.ru.nl
-> password
ssh -J lilo.science.ru.nl cn99 
```
Create slurm job file (`.sh` file)
```
#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:2
#SBATCH --time=20:00:00

source pythesis/bin/activate
wandb login 
python basic_run.py
```
Run the slurm job 
```
sbatch -w cn47 basicrun.sh 
```
To watch the queue on the servers, or to cancel the slurm job, use the following commands,
```
squeue 
scancel JOBID
```
