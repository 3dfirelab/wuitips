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



