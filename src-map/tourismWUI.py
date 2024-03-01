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

    importlib.reload(tools)

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
    indir_clc = '{:s}FuelCategories-CLC/{:s}/'.format(dir_data,continent)
    idxclc = range(1,7)
    fuelCat_all = []
    for iv in idxclc:
        fuelCat_ = gpd.read_file(indir_clc+'fuelCategory{:d}.geojson'.format(iv))
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

        if 'spotsall' in os.path.basename(spotsFile) : continue

        name_ = '-'.join(os.path.basename(spotsFile).split('.')[0].split('-')[1:])

        print('load tourism spot {:s}...'.format(name_), end='')
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
        #WUI['IDSpot'] =  '{:s}-{:6d}'.format(name_,WUI['IDSpot'])

        # Define a function that takes a row and returns a value based on 'input_column'
        def custom_function(row):
            return '{:s}-{:06d}'.format(name_,int(row['IDSpot']))

        # Create 'output_column' by applying the function
        WUI['IDSpot'] = WUI.apply(custom_function, axis=1)
        spots['IDSpot'] = spots.apply(custom_function, axis=1)
        
        WUI['area_ha'] = WUI['geometry'].area/ 10**4

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


    #to save clc for the zone
    forest = pd.concat(fuelCat_all[:3])
    agri = pd.concat(fuelCat_all[3:])
    minx,miny,maxx,maxy = WUI_tot.total_bounds

    minx -= 1000
    miny -= 1000
    maxx += 1000
    maxy += 1000

    forest.cx[minx:maxx,miny:maxy].to_file(indir_clc+'forest.geojson',driver='GeoJSON')
    agri.cx[minx:maxx,miny:maxy].to_file(indir_clc+'agriculture.geojson',driver='GeoJSON')

