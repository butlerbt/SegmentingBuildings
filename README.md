# SegmentingBuildings
This project uses deep learning models and computer vision to segment and mask building footprints from aerial imagery.

### 
Developing robust models that can automate feature identification and mapping from aerial imagery could reduce the burdensome task of having GIS technicians do this work, thereby allowing organizations to keep more up to date records and potentially more accurate records by reducing the human error of the technicians. In particular this has powerful resonance in quickly developing regions like many of the African countries where growth is fast, often times unmanaged, and tough and expensive to monitor. 


#### This README.pdf file will serve as a roadmap to this repository. The repository is open and available to the public.

### Directories and files to be aware of:

### • `solaris` conda environment

This project relies on you using the [`environment.yml`](environment.yml) file to recreate the `solaris` conda environment. To do so, please run the following commands:

```bash
# create the solaris conda environment
conda env create -f environment.yml

# activate the solaris conda environment
conda activate solaris

# make solaris available to you as a kernel in jupyter
python -m ipykernel install --user --name solaris --display-name "Python (solaris)"
```

### • `.src` source code:

This project contains several .py modules in the `src/utilities` directory. Please use the following bash command to install the .src module:

``` bash
#install the .src modules
pip install -e .
```

### • A notebooks directory that contains multiple Jupyter Notebooks:
    1. `notebooks/preprocessing/tier1_batch_processing_colab.ipynb`

         This notebook performs the preproccesing of the data to ready it for feeding into a CNN. 
        

    2. `notebooks/modeling/model.ipynb`

        This notebook goes through several iterations of training a Resnet-34 model for this specific segmentation task. 
        
    3. `notebooks/infer/competition_predictions.ipynb`

        This notebook performans predictive segmentation on the test set of data. 

### • The raw data was downloaded from Driven Data's competition which can be downloaded by running the `get_data.sh` script in the `src/data/` directory 
 

### • A one-page .pdf memo summarizing my project written for non technical stakeholders can be found in the `/reports/memo.md`

### • A pdf of my side deck summarizing my project can be found `/reports/presentation.pdf`




## Methodology 
Using 40 aerial images of neighborhood/city scale extents, I train a Convultional Neural Network to segment building footpints. 

I first broke each of the large-extent aerial images and their corresponding labels into smaller tiles ranging from 128x128 pixel up to 2056x2056 pixel. 20% of the tiles from each large-scale zone were set aside as a validation set. 

Then the shapefile labels (georefernced polygons) needed to be converted from vector data into raster data. 

This left me with a dataset of about 30000 individual tiles with thier corresponding labels. To begin with, I trained my model on just 500 of these. I then progressed to training on the entire dataset, both at 128x128 and 256x256 resolution. 

I used a resnet-34 with a Unet architecture. I used dice accuracty coeficient as my metric, and flattened cross entropy loss as my loss function.  

## Results 

My model was able to peform reasonably well achieving an 88%  dice accuracy on the validation data, and creating inferences on 512x512 tiles in just a few seconds.

I deployed the model on a flask app which can be viewed in my FastMap- public repository. 



