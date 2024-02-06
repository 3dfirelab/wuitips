import numpy as np 
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import glob 
import cv2
import sys
from scipy import ndimage, stats, spatial
import sys
from shapely.geometry import box
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import importlib
import pdb 

#homebrewed
import tools
sys.path.append('../src-load/')
glc = importlib.import_module("load-glc-category")   


##########################
if __name__ == '__main__':
##########################
    crs_here = 'epsg:3035'
    distgroup = 1.e4
    dir_data = '/mnt/dataMoor/WUITIPS/'
    dirin = dir_data+'TourismSpots/'

    sys.stdout.flush()
    continent = 'europe'
    indir = '{:s}FuelCategories-CLC/{:s}/'.format(dir_data,continent)
    idxclc = range(1,7)

    dbox = 1000.
    ptdx = 100
    for iv in idxclc:
        print('fuel class = {:d}'.format(iv))
        gdf = gpd.read_file(indir+'fuelCategory{:d}.geojson'.format(iv))
        gdf = tools.add_AI2gdf(gdf,ptdx,dbox)
        fuelCat.to_file(indir+'fuelCategory{:d}.geojson'.format(iv), driver='GeoJSON')
        
