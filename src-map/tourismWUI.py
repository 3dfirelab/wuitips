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
import os 
import pdb 

#homebrewed
import tools
sys.path.append('../src-load/')
glc = importlib.import_module("load-glc-category")



##########################
def buildWUI(WUI, iv, fuelCat, spots):

    importlib.reload(glc)
   
    bufferDistVegCat = [2400,1200,800,600,480,400,343]
    bb = 10.e3 
    nbregroup = spots['group'].max() +1
    #print('fuelCat{:d} - nbre group = {:d}'.format(iv,nbregroup))
    if  type(nbregroup) is not np.int64: pdb.set_trace()
    for ig in range(0, nbregroup):
        print('fuelCat{:d} - group {:d}/{:d} ... '.format(iv,ig,nbregroup),end='\r')
        sys.stdout.flush()
        spots_ = spots.loc[spots['group']==ig]
        #print(spots_.shape)
        xmin, ymin, xmax, ymax = spots_.total_bounds
        
        if type(fuelCat) is gpd.geodataframe.GeoDataFrame:
            fuelCat_ = fuelCat.cx[xmin-bb:xmax+bb, ymin-bb:ymax+bb]
        else: 
            indir = '{:s}CLC/'.format(get_dirData())
            to_latlon = pyproj.Transformer.from_crs(spots_.crs, 'epsg:4326')
            if np.abs(xmax-xmin) < 30.e3: bbx = 15.e3 
            else: bbx = bb
            if np.abs(ymax-ymin) < 30.e3: bby = 15.e3 
            else: bby = bb

            #lowerCorner = to_latlon.transform(ymin-bbx, xmin-bbx )
            #upperCorner = to_latlon.transform(ymax+bby, xmax+bby )
            
            lowerCorner = to_latlon.transform(xmin-bbx, ymin-bbx )
            upperCorner = to_latlon.transform(xmax+bby, ymax+bby )

            fuelCat_ = glc.clipped_fuelCat_gdf(indir, iv, spots_.crs, lowerCorner[1], lowerCorner[0], upperCorner[1], upperCorner[0])
            if fuelCat_ is None:
                continue
            else: 
                fuelCat_  =  tools.add_AI2gdf(fuelCat_,ptdx=100,dbox=1000,PoverA=0.05)
   
        for iai in range(3):
            if iai == 0: 
                fuelCat__ = fuelCat_[(fuelCat_['AI']>0.9)                      ]
            elif iai == 1: 
                fuelCat__ = fuelCat_[(fuelCat_['AI']>0  )&(fuelCat_['AI']<=0.9)]
            elif iai == 2: 
                fuelCat__ = fuelCat_[(fuelCat_['AI']<=0 )                      ]  # to update to pass it to ==0. need PoverA update. 
        
            vegCat = iv + iai -1
            bufferDistVegCat_ = bufferDistVegCat[vegCat]

            spots__ = spots_.copy()
            spots__['geometry'] = spots_.geometry.apply(lambda g: g.buffer(bufferDistVegCat_))
       
            if len(fuelCat__)>0:
                WUI_ = gpd.overlay(fuelCat__, spots__, how = 'intersection', keep_geom_type=False)
                pdb.set_trace()
            else: 
                continue 

            if WUI is None: 
                WUI = WUI_
            elif WUI_.shape[0]>0:
                WUI = pd.concat([WUI, WUI_])
   
        #ax = plt.subplot(111)
        #WUI.plot(ax=ax)
        #spots_.plot(ax=ax,color='k')
        #plt.show()
        #pdb.set_trace()

    return WUI



if __name__ == '__main__':

    crs_here = 'epsg:3035'
    distgroup = 1.e4
    dirin  = './TourismSpots/'
    dirout = './WUI/'


    print('load clc ...', end='')
    sys.stdout.flush()
    dir_data = '/mnt/dataEuropa/WII/'
    continent = 'europe'
    indir = '{:s}FuelCategories-CLC/{:s}/'.format(dir_data,continent)
    idxclc = range(1,6)
    fuelCat_all = []
    for iv in idxclc:
        fuelCat_ = gpd.read_file(indir+'fuelCategory{:d}.geojson'.format(iv))
        fuelCat_ = fuelCat_.to_crs(crs_here)
        fuelCat_all.append(fuelCat_)
    print(' done')


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
            WUI = buildWUI(WUI, iv, fuelCat_all[iv-1], spots)

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


