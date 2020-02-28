#install many of the necessary libraries. Will typically get an error regarding "pillow". Just re run cell
import solaris as sol
import numpy as np
import pandas as pd
import geopandas as gpd
import descartes
from matplotlib import pyplot as plt
from pathlib import Path
import rasterio
import os
import supermercado
import glob


from rio_tiler import main as rt_main

import mercantile
from rasterio.transform import from_bounds
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

import skimage
from tqdm import tqdm

# def import_data(zone, region, train_tier):
#     """ 
#     Imports the COG aerial imagery and labled geojson
#     returns: label_df, geotif pathname
#     """
#     geojson = f'../../data/raw/{train_tier}/{region}/{zone}-labels/{zone}.geojson'
#     geotif = f'../../data/raw/{train_tier}/{region}/{zone}/{zone}.tif'
#     label_df = gpd.read_file(geojson)
#     geotif = geotif
#     return label_df, geotif

def import_data_colab(label_file_df, tif_file_df, iteration):
    """ 
    Imports the COG aerial imagery and labled geojson
    returns: label_df, geotif pathname
    """
    i = iteration

    geotif = tif_file_df['tif_file_path'][i]
    region = tif_file_df['region'][i]
    zone = tif_file_df['zone'][i]
    geojson = label_file_df.loc[label_file_df['zone']==zone]['label_file_path'].values[0]

    label_df = gpd.read_file(geojson)
    return label_df, geotif, region, zone

def make_processed_directories_colab(zone, region, zoom_level = 19, image_size = 256):
    """
    make the directories to store processed/output data
    """
    os.system(f'mkdir drive/My\ Drive/seg_building_foots/data/processed/images')
    os.system(f'mkdir drive/My\ Drive/seg_building_foots/data/processed/masks')
    img_path = f'drive/My\ Drive/seg_building_foots/data/processed/images'
    mask_path = f'drive/My\ Drive/seg_building_foots/data/processed/masks'
    return img_path, mask_path

# def make_processed_directories(zone, region, zoom_level = 19, image_size = 256):
#     """
#     make the directories to store processed/output data
#     """
#     os.system(f'mkdir ../../data/processed/images-{image_size}-{region}-{zone}-{zoom_level}')
#     os.system(f'mkdir ../../data/processed/masks-{image_size}-{region}-{zone}-{zoom_level}')
#     img_path = f'../../data/processed/images-{image_size}-{region}-{zone}-{zoom_level}'
#     mask_path = f'../../data/processed/masks-{image_size}-{region}-{zone}-{zoom_level}'
#     return img_path, mask_path
    
def file_path_df_colab(train_tier):
    """
    returns a df containing all filepaths for both labels and geotifs
    """
    # Prep list of filepaths
    ims_filepaths = glob.glob(f'train_tier_{train_tier}/*/*/*.tif')
    label_filepaths = glob.glob(f'train_tier_{train_tier}/*/*/*.geojson')
    print("Found ", len(ims_filepaths), "images")
    print("Found", len(label_filepaths),"labels")

    tif_filepath_df = pd.DataFrame({
        'tif_file_path':ims_filepaths,
        'region':[t.split('/')[1] for t in ims_filepaths],
        'zone':[x.split('/')[2].split('-')[0] for x in ims_filepaths]
    
    })

    label_filepath_df = pd.DataFrame({
        'label_file_path':label_filepaths,
        'region':[t.split('/')[1] for t in label_filepaths],
        'zone':[x.split('/')[2].split('-')[0] for x in label_filepaths]    
    })
    return tif_filepath_df, label_filepath_df


def make_filepath_df(img_path):
    """
    use glob to put all of the file names, train and validation,
    into a dataframe for easy access. 
    """
    
    ims = glob.glob(f'{img_path}/*.png')
    filepath_df = pd.DataFrame({
        'img_path':ims,
        'mask_path':[im.replace('images', 'masks').replace('.png', '_mask.png') for im in ims]
    })

    return filepath_df

# def make_filepath_df(region, zone, zoom_level = 19, image_size = 256):
#     """
#     use glob to put all of the file names, train and validation,
#     into a dataframe for easy access. 
#     """
    
#     ims = glob.glob(f'../../data/processed/images-{image_size}-{region}-{zone}-{zoom_level}/*.png')
#     filepath_df = pd.DataFrame({
#         'img_path':ims,
#         'mask_path':[im.replace('images', 'masks').replace('.png', '_mask.png') for im in ims]
#     })

#     return filepath_df


# def burn_tiles(region, zone, train_tier = 1, zoom_level = 19):
#     """
#     Uses supermercado bash command to burn tiles across region/zone
#     at set zoom level. 

#     input: zoom_level = int, slippy map tile zooom level
#     region: the region to be burned
#     zone: the zone to be burned
#     train_tier = training tier the zone and region belong to

#     output: geojson with tile geometry with relative path name: 
#             'tiles{region}{zone}.geojson'
#     """
    
#     os.system(f'cat ../../data/raw/train_tier_{train_tier}/{region}/{zone}/{zone}.json | supermercado burn {zoom_level} | mercantile shapes | fio collect > ../../data/raw/train_tier_{train_tier}/{region}/{zone}/tiles_{region}_{zone}_{zoom_level}.geojson')
#     os.system(f'echo done with {region}_{zone}_{zoom_level}')

def burn_tiles_colab(region, zone, train_tier = 1, zoom_level = 19):
    """
    Uses supermercado bash command to burn tiles across region/zone
    at set zoom level. 

    input: zoom_level = int, slippy map tile zooom level
    region: the region to be burned
    zone: the zone to be burned
    train_tier = training tier the zone and region belong to

    output: geojson with tile geometry with relative path name: 
            'tiles{region}{zone}.geojson'
    """
    
    os.system(f'cat train_tier_{train_tier}/{region}/{zone}/{zone}.json | supermercado burn {zoom_level} | mercantile shapes | fio collect > train_tier_{train_tier}/{region}/{zone}/tiles_{region}_{zone}_{zoom_level}.geojson')
    os.system(f'echo done with {region}_{zone}_{zoom_level}')

# def load_tile_geojson(region, zone, train_tier = 1, zoom_level = 19, visualize = False):
#     """
#     loads into a gpd dataframe and visualizes the supermercado created tile geojson

#     returns gdf
#     """

#     tiles_gdf = gpd.read_file(f'../../data/raw/train_tier_{train_tier}/{region}/{zone}/tiles_{region}_{zone}_{zoom_level}.geojson')
#     if visualize:
#         tiles_gdf.plot(figsize=(10,10), color='grey', alpha=0.5, edgecolor='red')
    
#     return tiles_gdf

def load_tile_geojson_colab(region, zone, train_tier = 1, zoom_level = 19, visualize = False):
    """
    loads into a gpd dataframe and visualizes the supermercado created tile geojson

    returns gdf
    """

    tiles_gdf = gpd.read_file(f'train_tier_{train_tier}/{region}/{zone}/tiles_{region}_{zone}_{zoom_level}.geojson')
    if visualize:
        tiles_gdf.plot(figsize=(10,10), color='grey', alpha=0.5, edgecolor='red')
    
    return tiles_gdf

def train_validation_split(tiles_df, valid_set_size = .2, visualize = False):
    """
    Splits the burned tiles into training and validation sets

    imput: gdf of tiles 

    output: gdf of tiles split between training and validation sets
    """
    val_tile_indexes = np.random.choice(len(tiles_df), size = round(valid_set_size*len(tiles_df)))
    tiles_df['dataset']='training'
    tiles_df.loc[val_tile_indexes, 'dataset'] = 'validation'

    if visualize:
        fig, ax = plt.subplots(figsize=(10,10))
        tiles_df.loc[tiles_df['dataset']=='training'].plot(ax=ax, color='grey', alpha=0.5, edgecolor='red')
        tiles_df.loc[tiles_df['dataset']=='validation'].plot(ax=ax, color='grey', alpha=0.5, edgecolor='blue')
    return tiles_df

def reformat_xyz(tile_gdf):
    """
    Convert 'id' string to list of ints for z,x,y
    """
    tile_gdf['xyz'] = tile_gdf.id.apply(lambda x: x.lstrip('(,)').rstrip('(,)').split(','))
    tile_gdf['xyz'] = [[int(q) for q in p] for p in tile_gdf['xyz']]
    return tile_gdf



def explode(gdf):
    """    
    # https://gis.stackexchange.com/questions/271733/geopandas-dissolve-overlapping-polygons
    # https://nbviewer.jupyter.org/gist/rutgerhofste/6e7c6569616c2550568b9ce9cb4716a3

    Will explode the geodataframe's muti-part geometries into single 
    geometries. Each row containing a multi-part geometry will be split into
    multiple rows with single geometries, thereby increasing the vertical size
    of the geodataframe. The index of the input geodataframe is no longer
    unique and is replaced with a multi-index. 

    The output geodataframe has an index based on two columns (multi-index) 
    i.e. 'level_0' (index of input geodataframe) and 'level_1' which is a new
    zero-based index for each single part geometry per multi-part geometry
    
    Args:
        gdf (gpd.GeoDataFrame) : input geodataframe with multi-geometries
        
    Returns:
        gdf (gpd.GeoDataFrame) : exploded geodataframe with each single 
                                 geometry as a separate entry in the 
                                 geodataframe. The GeoDataFrame has a multi-
                                 index set to columns level_0 and level_1
        
    """
    gs = gdf.explode()
    gdf2 = gs.reset_index().rename(columns={0: 'geometry'})
    gdf_out = gdf2.merge(gdf.drop('geometry', axis=1), left_on='level_0', right_index=True)
    gdf_out = gdf_out.set_index(['level_0', 'level_1']).set_geometry('geometry')
    gdf_out.crs = gdf.crs
    return gdf_out


def cleanup_invalid_geoms(all_polys):
    """
    input: gdf.geometry column consisting of multiple rows of sinlge and multiple polygons
    returns: gdf of polygon geometrys that have been cleanedup 
    """
    all_polys_merged = gpd.GeoDataFrame()
    all_polys_merged['geometry'] = gpd.GeoSeries(cascaded_union([p.buffer(0) for p in all_polys]))

    gdf_out = explode(all_polys_merged)
    gdf_out = gdf_out.reset_index()
    gdf_out.drop(columns=['level_0','level_1'], inplace=True)
    all_polys = gdf_out['geometry']
    return all_polys


def get_specific_tile(idx, tiles_gdf):
    """
    Gets specific tile from the supermercado burned tiles that matches the clipped raster image tile

    input: tiles_gdf post train/validation split  
    idx: specific index number of tile
    """
    tile_poly = tiles_gdf.iloc[idx]['geometry']
    # print(tile_poly.bounds)
    return tile_poly

def clip_singletile_polys(idx, tiles_gdf, all_polys_series, tile_size):
    """
    Clips and visualizes the polygon labels for a single tile
    input: 

    idx: the index number of supermercado tile to clip polygon to 
    tiles_gdf : gdf continaing supermercado tiles
    all_polys_series : series containing polygon geometries 
    tile_size : size of supermercado tile 

    outputs:
    visualizes clipped polygons
    returns gdf of polygons within the specified idx tile
    """
    tile_poly = get_specific_tile(idx = idx, tiles_gdf=tiles_gdf)
    # tfm = from_bounds(*tile_poly.bounds, tile_size, tile_size) 
    cropped_polys = [poly for poly in all_polys_series if poly.intersects(tile_poly)]
    cropped_polys_gdf = gpd.GeoDataFrame(geometry=cropped_polys, crs={'init': 'epsg:4326'})
    cropped_polys_gdf.plot()
    return cropped_polys_gdf



def save_tile_img(tif, xyz, dataset, tile_size, region, zone, save_path, display=False):
    """
    clips COG to specific supermercado burned tile and saves the tile to the save_path
    """
    
    prefix = f'{region}{zone}{dataset}_'
    x,y,z = xyz
    tile, mask = rt_main.tile(tif, x,y,z, tilesize=tile_size)
    if display: 
        plt.imshow(np.moveaxis(tile,0,2))
        plt.show()
    
    skimage.io.imsave(f'{save_path}/{prefix}{z}_{x}_{y}.png',np.moveaxis(tile,0,2), check_contrast=False) 

def tfm(tile_polygon, tile_size):
    """
    Returns the Affine transformation matrix. 

    Affine transformation is a linear mapping method that preserves points, straight lines, and planes. 
    Sets of parallel lines remain parallel after an affine transformation. The affine transformation technique 
    is typically used to correct for geometric distortions or deformations that occur with non-ideal camera angles.
    """

    tfm = from_bounds(*tile_polygon.bounds, tile_size, tile_size)
    return tfm


def burn_mask(cropped_polys_gdf, tfm, tile_size, channels = 3):
    """
    Use solaris to create a pixel mask¶
    Creates our corresponding 3-channel RGB mask by passing the cropped polygons to solaris' df_to_px_mask function.

    1st (Red) channel represent building footprints,
    2nd (Green) channel represent building boundaries (visually looks yellow on the RGB mask display because the pixels overlap red and green+red=yellow),
    3rd (Blue) channel represent close contact points between adjacent buildings

    see: https://solaris.readthedocs.io/en/latest/tutorials/notebooks/api_masks_tutorial.html
    """
    
    

    if channels == 3:
        fbc_mask = sol.vector.mask.df_to_px_mask(df=cropped_polys_gdf,
                                                channels=['footprint', 'boundary', 'contact'],
                                                affine_obj=tfm, shape=(tile_size,tile_size),
                                                boundary_width=5, boundary_type='inner', contact_spacing=5, meters=True)
    elif channels == 1:
        fbc_mask = sol.vector.mask.footprint_mask(df=cropped_polys_gdf,
                                                    affine_obj=tfm, shape=(tile_size,tile_size),
                                                    )
    return fbc_mask


def save_tile_mask(label_poly_series, tile_poly, xyz, tile_size, dataset, region, zone, save_path, channels = 3, display=False):
    """
    clips the label polygons to the extent of a supermercado tile, uses solaris to burn a mask from the polygons, and 
    saves the burned mask to the directory
    """
     
    

    prefix = f'{region}{zone}{dataset}_'
    x,y,z = xyz
    tfm = from_bounds(*tile_poly.bounds, tile_size, tile_size) 
  
    cropped_polys = [poly for poly in label_poly_series if poly.intersects(tile_poly)]
    cropped_polys_gdf = gpd.GeoDataFrame(geometry=cropped_polys, crs={'init': 'epsg:4326'})
    
    fbc_mask = burn_mask(cropped_polys_gdf, tfm, tile_size, channels)
        # fbc_mask = sol.vector.mask.df_to_px_mask(df=cropped_polys_gdf,
        #                                     channels=['footprint', 'boundary', 'contact'],
        #                                     affine_obj=tfm, shape=(tile_size,tile_size),
        #                                     boundary_width=5, boundary_type='inner', contact_spacing=5, meters=True)
  
    if display: 
        plt.imshow(fbc_mask); plt.show()
  
    skimage.io.imsave(f'{save_path}/{prefix}{z}_{x}_{y}_mask.png',fbc_mask, check_contrast=False) 

def batch_save_tile_img(tiles_gdf, tif, tile_size, region, zone, save_path, display=False):
    """
    batch clips the entire COG

    cool things to remember about this loop:
    1. tqdm creats a progress bar for the loop
    2. .iterrows makes it easy to iterate through a Dataframe

    """
    for idx, tile in tqdm(tiles_gdf.iterrows()):
        dataset = tile['dataset']
        save_tile_img(tif, tile['xyz'], dataset, tile_size, region, zone, save_path, display=False)

def batch_save_tile_mask(tiles_gdf, label_poly_series, tile_size, region, zone, save_path, channels=3, display=False):
    """
    Batch clips, burns, and saves the entire set of polygon labels  

    inputs:
    tiles_gdf : gdf of supermercado tiles
    label_poly_series : series or geodaframe['column'] consisting of polygon geometries 
    tile_size : size of supermercado tile and output slippy tile
    zone :  zone of study, for naming convention 
    region : region of study, for naming convention
    channels : bands to be burned into mask. 1 =  only building footprint; 3 = footprint, boundary, and contact
    display : visualize each mask

    outputs: 
    saves burned masks in relative directory

    cool things to remember about this loop:
    1. tqdm creats a progress bar for the loop
    2. .iterrows makes it easy to iterate through a Dataframe
    """
     
    import warnings; warnings.simplefilter('ignore')

    for idx, tile in tqdm(tiles_gdf.iterrows()):
        dataset = tile['dataset']
        tile_poly = get_specific_tile(idx, tiles_gdf)
        save_tile_mask(label_poly_series, tile_poly, tile['xyz'], tile_size, dataset,
                         region, zone, save_path, channels, display)


