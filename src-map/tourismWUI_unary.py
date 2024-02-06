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
import geopandas as gpd
import importlib
import pygeos as pg
from shapely.validation import make_valid, explain_validity
import shapely


if __name__ == '__main__':

    '''
    need geo env with shapely > version 2.
    '''

    dirinWUI = './WUI/'
    dirinSpot = './TourismSpots/'


    WUI_ = gpd.read_file(dirinWUI+'WUIall.geojson')
    spots_ = gpd.read_file(dirinSpot+'spotsall.geojson')
    
    round = np.vectorize(lambda geom: pg.apply(geom, lambda g: g.round(3)))
    WUI_.geometry = round(WUI_.geometry.values.data)
    
    WUI_.geometry = WUI_.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)

    for index, row in WUI_[(WUI_.geom_type != 'Polygon') & (WUI_.geom_type!='MultiPolygon')].iterrows():
        #with warnings.catch_warnings(record=True) as w:
        WUI_.at[index,'geometry'] =  WUI_[index:index+1].geometry.buffer(1.e-6).unary_union 
        
    #keep single overlap
    # define a function that rounds the coordinates of every geometry in the array
    ##MERDEtmp_ = round(WUI_.geometry.values.data)
    WUI_['geometry'] = shapely.set_precision(WUI_.geometry.values, 1e-2)
    WUI_ = gpd.GeoDataFrame(geometry=[WUI_.unary_union], crs=WUI_.crs).explode( index_parts=False ).reset_index( drop=True )
    #WUI_.geometry = WUI_.buffer(-0.01)

    #to force remove indus from WUI
    spots_['geometry'] = shapely.set_precision(spots_.geometry.values, 1e-2)
    WUI_ = WUI_.overlay(spots_, how = 'difference', keep_geom_type=False)

    WUI_['area_ha'] = WUI_.area * 1.e-4

    WUI_.to_file(dirinWUI+'WUI-unary_union.geojson', driver='GeoJSON')
    print ('done')
    sys.stdout.flush()


