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
import os
os.environ['USE_PYGEOS'] = '0'
import pandas as pd
import geopandas as gpd
import importlib
import pdb 
import warnings
import tools 

# Convert all warnings to errors
warnings.filterwarnings("error")

if __name__ == '__main__':


    crs_here = 'epsg:3035'
    dir_data = '/mnt/dataEuropa/WUITIPS/'
    dirinSpot = dir_data+'TourismSpots-EU/'
    dirinBorders = dir_data+'Borders/'
    dirinWUI = dir_data+'WUI/'
    diroutWUI = dir_data+'WUI/PerCountry/'

    tools.ensure_dir(diroutWUI)

    borders = gpd.read_file(dirinBorders+'NUTS_RG_01M_2021_4326.geojson')
    borders = borders.to_crs(crs_here)
    countries = borders[borders['LEVL_CODE']==0]


    wui_all = gpd.read_file(dirinWUI+'WUIall.geojson')
    spots_all = gpd.read_file(dirinSpot+'spotsall_withflag.geojson')

    for ii, row in countries.iterrows():
        
        #if row['CNTR_CODE']!= 'IT': continue
        print(row['CNTR_CODE'])
        country=gpd.GeoDataFrame([row])
        country['geometry']=row['geometry']
        country.crs=borders.crs
        #ax = plt.subplot(111)
        #country.plot(ax=ax)
        

        intersectionW_gdf = gpd.sjoin(wui_all,  country, how="inner", predicate="intersects")
        intersectionS_gdf = gpd.sjoin(spots_all,country, how="inner", predicate="intersects")

        
        intersectionS_gdf.to_file(diroutWUI+'{:s}_{:s}.geojson'.format('spot', row['CNTR_CODE']))
        intersectionW_gdf.to_file(diroutWUI+'{:s}_{:s}.geojson'.format('wui', row['CNTR_CODE']))
        #intersection_gdf.plot(ax=ax,color='orange')
        #plt.show()



