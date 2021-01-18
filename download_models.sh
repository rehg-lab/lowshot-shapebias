#!/usr/bin/env bash

# download models
wget https://dl.dropbox.com/s/7uxv61iq1afkxpo/pretrained_models.tar

# untar data
tar -xf pretrained_models.tar 

# remove tarballs
rm pretrained_models.tar
