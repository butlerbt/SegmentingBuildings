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

def import_data(zone, region, train_tier):
    """ 
    Imports the COG aerial imagery and labled geojson
    returns: label_df, geotif pathname
    """
    geojson = f'../../data/raw/{train_tier}/{region}/{zone}-labels/{zone}.geojson'
    geotif = f'../../data/raw/{train_tier}/{region}/{zone}/{zone}.tif'
    label_df = gpd.read_file(geojson)
    geotif = geotif
    return label_df, geotif


def make_processed_directories(zone, region, zoom_level = 19, image_size = 256):
    """
    make the directories to store processed/output data
    """
    os.system(f'mkdir ../../data/processed/images-{image_size}-{region}-{zone}-{zoom_level}')
    os.system(f'mkdir ../../data/processed/masks-{image_size}-{region}-{zone}-{zoom_level}')
    img_path = f'../../data/processed/images-{image_size}-{region}-{zone}-{zoom_level}'
    mask_path = f'../../data/processed/masks-{image_size}-{region}-{zone}-{zoom_level}'
    return img_path, mask_path
    



def make_filepath_df(region, zone, zoom_level = 19, image_size = 256):
    """
    use glob to put all of the file names, train and validation,
    into a dataframe for easy access. 
    """
    
    ims = glob.glob(f'../../data/processed/images-{image_size}-{region}-{zone}-{zoom_level}/*.png')
    filepath_df = pd.DataFrame({
        'img_path':ims,
        'mask_path':[im.replace('images', 'masks').replace('.png', '_mask.png') for im in ims]
    })

    return filepath_df


def burn_tiles(region, zone, train_tier = 1, zoom_level = 19):
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
    
    os.system(f'cat ../../data/raw/train_tier_{train_tier}/{region}/{zone}/{zone}.json | supermercado burn {zoom_level} | mercantile shapes | fio collect > ../../data/raw/train_tier_{train_tier}/{region}/{zone}/tiles_{region}_{zone}_{zoom_level}.geojson')
    os.system(f'echo done with {region}_{zone}_{zoom_level}')


def load_tile_geojson(region, zone, train_tier = 1, zoom_level = 19, visualize = False):
    """
    loads into a gpd dataframe and visualizes the supermercado created tile geojson

    returns gdf
    """

    tiles_gdf = gpd.read_file(f'../../data/raw/train_tier_{train_tier}/{region}/{zone}/tiles_{region}_{zone}_{zoom_level}.geojson')
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
    print(tile_poly.bounds)
    return tile_poly


def save_tile_img(tif_path, xyz, dataset, tile_size, zone, region, save_path, display=False):
    """
    clips COG to specific supermercado burned tile and saves the tile to the save_path
    """
    
    prefix = f'{region}{zone}{dataset}_'
    x,y,z = xyz
    tile, mask = rt_main.tile(tif_path, x,y,z, tilesize=tile_size)
    if display: 
        plt.imshow(np.moveaxis(tile,0,2))
        plt.show()
    
    skimage.io.imsave(f'{save_path}/{prefix}{z}_{x}_{y}.png',np.moveaxis(tile,0,2), check_contrast=False) 



def save_tile_mask(labels_poly, tile_poly, xyz, tile_size, dataset, zone, region, save_path, display=False):
    """
    clips the label polygons to the extent of a supermercado tile, uses solaris to burn a mask from the polygons, and 
    saves the burned mask to the directory
    """

    prefix = f'{region}{zone}{dataset}_'
    x,y,z = xyz
    tfm = from_bounds(*tile_poly.bounds, tile_size, tile_size) 
  
    cropped_polys = [poly for poly in labels_poly if poly.intersects(tile_poly)]
    cropped_polys_gdf = gpd.GeoDataFrame(geometry=cropped_polys, crs={'init': 'epsg:4326'})
  
    fbc_mask = sol.vector.mask.df_to_px_mask(df=cropped_polys_gdf,
                                         channels=['footprint', 'boundary', 'contact'],
                                         affine_obj=tfm, shape=(tile_size,tile_size),
                                         boundary_width=5, boundary_type='inner', contact_spacing=5, meters=True)
  
    if display: 
        plt.imshow(fbc_mask); plt.show()
  
    skimage.io.imsave(f'{save_path}/{prefix}{z}_{x}_{y}_mask.png',fbc_mask, check_contrast=False) 

def batch_save_tile_img(tiles_gdf, tif_path, xyz, data_set, tile_size, zone, region, save_path, display=False):
    """
    batch clips the entire COG

    cool things to remember about this loop:
    1. tqdm creats a progress bar for the loop
    2. .iterrows makes it easy to iterate through a Dataframe

    """
    for idx, tile in tqdm(tiles_gdf.iterrows()):
        dataset = tile['dataset']
        save_tile_img(tif_path, tile['xyz'], data_set, tile_size, zone, region, save_path, display=False)



