from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import skimage

def visualize_burned_cells(tiles_gdf, label_gdf):
    """
    Displays the supermercado burned cells over the polygon labels.

    input:
    tiles_gdf: geodataframe consisting of tile geometries 
    """

    fig, ax = plt.subplots(figsize=(10,10))
    tiles_gdf[tiles_gdf['dataset'] == 'training'].plot(ax=ax, color='grey', edgecolor='red', alpha=0.3)
    tiles_gdf[tiles_gdf['dataset'] == 'validation'].plot(ax=ax, color='grey', edgecolor='blue', alpha=0.3)
    label_gdf.plot(ax=ax)




def visualize_saved_tiles(start_range, end_range, df_filepaths):
    """
    Uses skimage to display tiles. Useful after batch processing to verifiy things worked as the should.
    inputs:
    start_range: index to start visualizing from
    end_range: index to end visualizing 
    df_filepaths: df that the start_range and end_range index number is referencing

    outputs:
    displays image and mask side-by-side
    """ 
    start_idx, end_idx = start_range, end_range
    for i in range(start_idx, end_idx):
        image, mask = df_filepaths.iloc[i]
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
        ax1.imshow(skimage.io.imread(image))
        ax2.imshow(skimage.io.imread(mask))
        plt.show()

