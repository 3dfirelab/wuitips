import sys
import os 
#os.environ['USE_PYGEOS'] = '0'
import pandas as pd 
import geopandas as gpd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyproj

if __name__ == '__main__':

    categories = np.arange(1,6)
    colorveg=['blue','green','orange','black','magenta']

    print('fuel clc')
    indir = '/mnt/dataEuropa/WII/CLC/u2018_clc2018_v2020_20u1_geoPackage/DATA/'
    outdir = './FuelCategories-CLC/europe/'
    
    #set forest and agricultural class according to https://clc.gios.gov.pl/doc/clc/CLC_Legend_EN.pdf
     
    fuelCatTag = []
    fuelCatTag.append(['312'])                                         #1 forest1: Coniferous forest 
    fuelCatTag.append(['313'])                                         #2 forest2: Mixed forest
    fuelCatTag.append(['311','321','323','324'])                       #3 forest3: Broad-leaved forest
    fuelCatTag.append(['322'])                                         #4 grass/shrub : Moors and heathland
    fuelCatTag.append(['411'])                                         #5 Inland marshes 
    fuelCatTag.append(['211,212,213,221,222,223,231,241,242,243,244']) #6 Agricultural

    clc = gpd.read_file(indir+'/U2018_CLC2018_V2020_20u1.gpkg')
    to_latlon = pyproj.Transformer.from_crs(clc.crs, 'epsg:4326')
    lowerCorner = to_latlon.transform(*clc.total_bounds[:2])
    upperCorner = to_latlon.transform(*clc.total_bounds[2:])
    print('bounding box')
    print(lowerCorner[::-1], upperCorner[::-1])
    
    for iv in categories:
        print('  fuel cat: {:d}'.format(iv),end='')

        condition =  (pd.Series([False]*len(clc)))
        for tag in fuelCatTag[iv-1]:
            condition |= (clc['Code_18']==tag)
        fuel = clc[condition]
        print(fuel.shape)
        
        fuel.to_file(outdir+'fuelCategory{:d}.geojson'.format(iv), driver="GeoJSON")

