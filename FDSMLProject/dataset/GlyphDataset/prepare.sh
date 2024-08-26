#!/usr/bin/env bash

wget http://iamai.nl/downloads/GlyphDataset.zip
unzip GlyphDataset.zip -d ../datasets/data
rm GlyphDataset.zip

rm -r ../datasets/prepared_data
mkdir ../datasets/prepared_data
python3 prepare_data.py