#!/usr/bin/env bash

mkdir data
cd data

# download modelnet
wget https://dl.dropbox.com/s/7lojtne1v6kgawe/ModelNet40-LS.tar

# download shapenet
wget https://dl.dropbox.com/s/4cp2cn0vx3ltw7p/ShapeNet55-LS.tar

# download toys
wget https://dl.dropbox.com/s/qjlc1smbg599xi7/TOYS4K.tar

# untar data
tar -xf ModelNet40-LS.tar
tar -xf ShapeNet55-LS.tar
tar -xf TOYS4K.tar

# remove tarballs
rm ModelNet40-LS.tar
rm ShapeNet55-LS.tar
rm TOYS4K.tar
