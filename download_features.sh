#!/usr/bin/env bash

# download features
wget https://www.dropbox.com/s/1lnu1is0uoj8w2e/features.tar
tar -xf features.tar

rm features.tar

mv ./features/mn_features/features ./data/ModelNet40-LS/
mv ./features/sn_features/features ./data/ShapeNet55-LS/
mv ./features/t_features/features ./data/TOYS4K/

rm -r ./features/
