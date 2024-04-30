#!/bin/bash
source ~/.bashrc
source activate stitch
alias python=python3


python train_dmlp.py dataset_name=pointmaze-umaze-v0 augment_data=True nclusters=40