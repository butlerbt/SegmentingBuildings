#Author: Brent Butler
#Date: 12/30/19
#Purpose: Download and organize data for Tanzania Water Pump Project

#!/bin/bash

echo "Start downloading data and documentation"

# bash function used to retrieve the absolute file path of a file as a string
# note: thank you to peterh's answer on SO 
#       https://stackoverflow.com/a/21188136
get_str_abs_filename() {
  # $1 : relative filename
  echo "'$(cd "$(dirname "$1")" && pwd)/$(basename "$1")'"
}

# download the Submission Format .csv file 
wget -P data/raw/ https://s3.amazonaws.com/drivendata/data/7/public/SubmissionFormat.csv

# download the the independent variables that need predictions (test set)
wget -P data/raw/ https://s3.amazonaws.com/drivendata/data/7/public/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv

# download the dependent variable (status_group) for each of the rows in training set values (training set labels)
wget -P data/raw/ https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv

# download the independent variables for the training set (training set predictors)
wget -P data/raw/ https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv

#download the Tanzania Country Geographic Data
wget -P data/raw/geographic/ https://data.humdata.org/dataset/451bdd28-d06d-46ea-91c0-2e081f884395/resource/b3d5fbd6-d5d9-47f6-9d23-f1b0bf25448a/download/tza_admbnda_adm0_20181019.zip

#download the Tanzania Region Geographic Data
wget -P data/raw/geographic/ https://data.humdata.org/dataset/451bdd28-d06d-46ea-91c0-2e081f884395/resource/55e9e1d3-6585-4e38-ae35-726875c1e8ed/download/tza_admbnda_adm1_20181019.zip

# download the Tanzania District Geographic Data
wget -P data/raw/geographic/ https://data.humdata.org/dataset/451bdd28-d06d-46ea-91c0-2e081f884395/resource/50fec0b7-ba83-43fb-bb1c-941e2dbc7a95/download/tza_admbnda_adm2_20181019.zip

# download the Tanzania Ward Geographic Data
wget -P data/raw/geographic/ https://data.humdata.org/dataset/451bdd28-d06d-46ea-91c0-2e081f884395/resource/930fb85d-f508-4e56-a7d6-0da021364861/download/tza_admbnda_adm3_20181019.zip

# download the Tanzania Population Density Raster Data
wget -P data/raw/geographic/ https://energydata.info/dataset/3979f68c-ccbd-4c88-8cdf-4897fda76a6b/resource/7fbd6649-fa93-43f9-bf8b-72c2e1450e69/download/tza-poptzapopmap15adjv2b.zip

# unpack the .zip files and place contents back into data/raw/ directory
unzip data/raw/geographic/tza_admbnda_adm0_20181019.zip -d data/raw/geographic
unzip data/raw/geographic/tza_admbnda_adm1_20181019.zip -d data/raw/geographic
unzip data/raw/geographic/tza_admbnda_adm2_20181019.zip -d data/raw/geographic
unzip data/raw/geographic/tza_admbnda_adm3_20181019.zip -d data/raw/geographic
unzip data/raw/geographic/tza-poptzapopmap15adjv2b.zip -d data/raw/geographic

#rename the csv files:
mv -iv data/raw/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv data/raw/train_set_labels.csv
mv -iv data/raw/4910797b-ee55-40a7-8668-10efd5c1b960.csv data/raw/train_set_values.csv
mv -iv data/raw/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv data/raw/test_set.csv



#rename the files that have now been downloaded:

#rename the test set:
mv -iv data/raw/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv data/raw/test_set.csv

#rename the training values labels
mv -iv data/raw/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv data/raw/train_set_labels.csv

#rename the training set values
mv -iv data/raw/4910797b-ee55-40a7-8668-10efd5c1b960.csv data/raw/train_set_values.csv



echo "Finished downloading data and documentation"
