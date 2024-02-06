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

#homebrewed
import tools
sys.path.append('../src-load/')
glc = importlib.import_module("load-glc-category")






if __name__ == '__main__':

    crs_here = 'epsg:3035'
    distgroup = 1.e4
    dir_data = '/mnt/dataMoor/WUITIPS/'
    dirin = dir_data+'TourismSpots/'
    dirout = dir_data+'WUI/'

    bufferDistVegCat = [200,200,200,50,50,50]
    flag_ai = False
    

    print('load clc ...', end='')
    sys.stdout.flush()
    dir_data = '/mnt/dataMoor/WUITIPS/'
    continent = 'europe'
    indir = '{:s}FuelCategories-CLC/{:s}/'.format(dir_data,continent)
    idxclc = range(1,7)
    fuelCat_all = []
    for iv in idxclc:
        fuelCat_ = gpd.read_file(indir+'fuelCategory{:d}.geojson'.format(iv))
        fuelCat_ = fuelCat_.to_crs(crs_here)
        fuelCat_all.append(fuelCat_)
    print(' done')

    if flag_ai: 
        if len(fuelCat_all) != len(bufferDistVegCat) + 3:
            print('bufferDistVegCat dimension is not matching the number of hazard cat = number of fuel cat + 3')
            sys.exit()
    else:
        if len(fuelCat_all) != len(bufferDistVegCat) :
            print('bufferDistVegCat dimension is not matching the number of fuel cat')
            sys.exit()



    WUI_tot = None
    spots_tot = None

    for spotsFile in glob.glob(dirin+'*.geojson'):
        
        if os.path.basename(spotsFile) == 'spotsall.geojson': continue

        print('load tourism spot {:s}...'.format(spotsFile), end='')
        spots = gpd.read_file(spotsFile)
        spots = spots.to_crs(crs_here)
        print(' done')
        
        WUI = gpd.GeoDataFrame(geometry=[])
        
        spots['area_ha'] = spots['geometry'].area/ 10**4
        #spots = spots[spots['area_ha']>1]
        print(' ', spots.shape)

        if spots.shape[0] == 0:
            continue

        elif spots.shape[0]>1:
            spots['group'] = tools.cluster_shapes_by_distance(spots, distgroup)

        else: 
            spots['group'] = 0 

        #print('nbre group :',spots['group'].max()+1)
        #spots['group'] = spots.group.astype(str)
        #spots.plot(column='group', legend=True)
        #plt.show()
        #sys.exit()

        for iv in idxclc:
            WUI = tools.buildWUI(WUI, iv, fuelCat_all[iv-1], spots, bufferDistVegCat)

        print ('WUI area_ha = ', WUI.area.sum()*1.e-4, '                 ' )
            
        if WUI_tot is None: 
            if WUI.shape[0]!=0:
                WUI_tot = WUI
        else:
            if WUI.shape[0]!=0:
                WUI_tot = pd.concat([WUI_tot, WUI])
 
        if spots_tot is None: 
            if spots.shape[0]!=0:
                spots_tot = spots
        else:
            if spots.shape[0]!=0:
                spots_tot = pd.concat([spots_tot, spots])

    WUI_tot.to_file(dirout+'WUIall.geojson',driver='GeoJSON')
    spots_tot.to_file(dirin+'spotsall.geojson',driver='GeoJSON')


