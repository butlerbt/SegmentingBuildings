#Author: Brent Butler
#Date: 12/30/19
#Purpose: Download and organize data for Capstone Project

#!/bin/bash

echo "Start downloading data and documentation"

# download the metadata file 
wget -P data/raw/ https://s3.amazonaws.com/drivendata/data/60/public/train_metadata.csv

# download th Submission Format file
wget -P data/raw/ https://s3.amazonaws.com/drivendata/data/60/public/submission_format.tgz

# download the high quality training data
wget -P data/raw/ https://drivendata-public-assets-eu.s3.eu-central-1.amazonaws.com/train_tier_1.tgz

#download the low quality training data
wget -P data/raw/geographic/ https://drivendata-public-assets.s3.amazonaws.com/train_tier_2.tgz

#download the Test Data
wget -P https://drivendata-public-assets.s3.amazonaws.com/test.tgz


# unpack the .tgz files and place contents back into data/raw/ directory
tar -xvzf data/raw/train_tier_1.tgz -C data/raw/
tar -xvzf data/raw/train_tier_2.tgz -C data/raw/
tar -xvzf data/raw/submission_format.tgz -C data/raw/
tar -xvzf data/raw/test.tgz -C data/raw/


#delete the .tgz archived files
rm data/raw/train_tier_1.tgz
rm data/raw/train_tier_2.tgz
rm data/raw/submission_format.tgz
rm data/raw/test.tgz 

echo "Finished downloading data and documentation"
