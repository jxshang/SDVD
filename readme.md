# SDVD: SDVD: Self-Supervised Dual-View Modeling of User and Cascade Dynamics for Information Diffusion Prediction

This project contains the source code for the work: **SDVD: Self-Supervised Dual-View Modeling of User and Cascade Dynamics for Information Diffusion Prediction**

**Authors**: Haoyu Xiong, Jiaxing Shang, Fei Hao, Dajiang Liu, Geyong Min

Please kindly give us a star if you find this code helpful.

## Table of Contents

- model.py: the overall model architecture
- static_graph_encoder.py: the code for Static Graph Learning module
- dynamic_graph_encoder.py: the code for Self-Supervised Dual-View Dynamic Modeling module
- transformer.py: the code for Transformer
- data.py: dataset preprocessing and graph construction
- epochs.py: the code for epoch execution
- run.py: the main procedure for model training and testing
- utils.py: contains utility functions

## Execution

python run.py --data_name android --n_interval 14     
python run.py --data_name christianity     
python run.py --data_name douban
python run.py --data_name twitter --hidden_dim 128   

## Requirements

see requirements.txt

## Contact

+ shangjiaxing@gmail.com (Jiaxing Shang)
+ yoyoking227@gmail.com (Haoyu Xiong)
