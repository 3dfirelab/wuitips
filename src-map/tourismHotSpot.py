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
import pygeos as pg
from shapely.validation import make_valid, explain_validity
import shapely


if __name__ == '__main__':

    '''
    need geo env with shapely > version 2.
    '''

    dir_data = '/mnt/dataEuropa/WUITIPS/'
    dirinSpot = dir_data+'TourismSpots-EU/'
    dirinWUI = dir_data+'WUI/'


    WUI = gpd.read_file(dirinWUI+'WUIall.geojson')
    spots = gpd.read_file(dirinSpot+'spotsall.geojson')
    
    WUI['area_ha'] = WUI['geometry'].area/ 10**4

    WUI.geometry = WUI.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)

    
    IDhotspot = np.unique(WUI[WUI['area_ha']>0.001]['IDSpot']) #>

    def setHotSpotFlag(row):
        if row['IDSpot'] in IDhotspot:
            return  1
        else: 
            return  0
    spots['flag_hotspot'] = spots.apply(setHotSpotFlag, axis=1)

    hotspots =  spots[spots['IDSpot'].isin(IDhotspot)]
    hotspots.to_file(dirinSpot+'hotspotsall.geojson',driver='GeoJSON')

    spots.to_file(dirinSpot+'spotsall_withflag.geojson',driver='GeoJSON')

    ax = plt.subplot(111)
    WUI.plot(ax=ax, alpha=0.3)
    spots.plot(ax=ax, color='k',alpha=.3)
    hotspots.plot(ax=ax, color='none', edgecolor='red')
    plt.show()
