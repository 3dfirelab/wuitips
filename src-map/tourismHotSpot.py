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


