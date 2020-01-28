import glob
import pandas as pd

def make_filepath_df(region, zone):
    """
    use glob to put all of the file names, train and validation,
    into a dataframe for easy access. 
    """


    ims = glob.glob(f'../../data/processed/images-256-{region}-{zone}/*.png')
    filepath_df = pd.DataFrame({
        'img_path':ims,
        'mask_path':[im.replace('images', 'masks').replace('.png', '_mask.png') for im in ims]
    })

    return filepath_df