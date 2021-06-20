#!/usr/bin/env bash

# download models
wget https://www.dropbox.com/s/dwh2u7ge6p4moax/pretrained_models.tar

# untar data
tar -xf pretrained_models.tar 

# remove tarballs
rm pretrained_models.tar
