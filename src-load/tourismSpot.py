import numpy as np 
import pyrosm 
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
import os
import glob 
import warnings
from shapely.errors import ShapelyDeprecationWarning
import pdb 

if __name__ == '__main__':

    crs_here = 'epsg:3035'
    dir_data = '/mnt/dataMoor/WUITIPS/'
    dirin = dir_data+'OSM/'
    dirout = dir_data+'TourismSpots/'

    for osmfilein in glob.glob(dirin+'*.pbf'):

        name = os.path.basename(osmfilein).split('-latest')[0]
        #if os.path.isfile('{:s}/tourism-{:s}.geojson'.format(dirout,name)): continue

        print(name)
        osm = pyrosm.OSM(filepath=osmfilein)
        custom_filter={'tourism': True}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)   
            warnings.simplefilter("ignore", ShapelyDeprecationWarning)   
            warnings.simplefilter("ignore", UserWarning)         
            tourism = osm.get_data_by_custom_criteria(custom_filter=custom_filter)
            tourism = tourism.to_crs(crs_here)
        print('tourism loaded')

        tourism_pt = tourism[(tourism.geom_type=='MultiPoint')|(tourism.geom_type=='Point')]
        #bounding_box = box(settlement_origin_x, settlement_origin_y-Ly, settlement_origin_x+Lx, settlement_origin_y)
        #tourism_pt = tourism_pt[tourism_pt.within(bounding_box)]
        
        #keep only polygon in tourism
        tourism = tourism[tourism.geom_type!='MultiLineString']
        tourism = tourism[tourism.geom_type!='LineString']
        tourism = tourism[tourism.geom_type!='MultiPoint']
        tourism = tourism[tourism.geom_type!='Point']
        tourism['origin_type'] = 'tourism'


        with warnings.catch_warnings():
           warnings.simplefilter("ignore", DeprecationWarning)   
           warnings.simplefilter("ignore", ShapelyDeprecationWarning)   
           warnings.simplefilter("ignore", UserWarning)  
           ShapelyDeprecationWarning
           buildings = osm.get_buildings()
        print('buildings loaded')
        buildings = buildings.to_crs(crs_here)
        buildings = buildings[(buildings.geom_type == 'Polygon' ) | (buildings.geom_type == 'MultiPolygon') ]

        def ckdnearest(gdA, gdB, distThreshold=1000, k=10):
            nA = np.array(list(gdA.geometry.apply(lambda x: (x.centroid.x, x.centroid.y))))
            nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
            btree = spatial.cKDTree(nB)
            dist, idx = btree.query(nA, k=k)
           
            gdA['idxPt'] = [list(row) for row in idx]
            gdA['distPt'] = [list(row) for row in dist]
            gdA['maxdistPt'] = [row.max() for row in dist]


            return gdA[gdA['maxdistPt'] < distThreshold]
        
        #print('run cdktree')
        #sys.stdout.flush()
        #buildings_tourism = ckdnearest(buildings,tourism_pt,distThreshold=1000)

        print('run sjoin')
        sys.stdout.flush()
        buildings_tourism = gpd.sjoin(buildings, tourism_pt, predicate='contains')
        
        buildings_tourism['origin_type'] = 'building'

        
        #merged
        tourism.geometry = tourism.geometry.buffer(-.01).buffer(.01)
        buildings_tourism.geometry = buildings_tourism.geometry.buffer(-.01).buffer(.01)

        common_columns = tourism.columns.intersection(buildings_tourism.columns)
        merged =  pd.concat([tourism[common_columns], buildings_tourism[common_columns]], ignore_index=True)
        merged['IDSpot'] = np.arange(len(merged))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ShapelyDeprecationWarning)   
            merged.geometry = merged.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)

        #remove overlap
        # Create a spatial index for the GeoDataFrame
        sindex = merged.sindex

        # List to hold indices of polygons to remove
        to_remove = []

        # Iterate over each polygon in the GeoDataFrame
        for index, polygon in merged.iterrows():
            # Find potential candidates that might contain the current polygon

            if len(polygon['geometry'].bounds) == 0: 
                to_remove.append(index)
                continue
                #pdb.set_trace()
            possible_matches_index = list(sindex.intersection(polygon['geometry'].bounds))
            possible_matches = merged.iloc[possible_matches_index]
           
            # Check for actual containment excluding the polygon itself
            flag_rm = False 
            possible_index_arr = []
            for possible_index, possible_polygon in possible_matches.iterrows():
                possible_index_arr.append(possible_index)
                if (index != possible_index) and (possible_index not in to_remove) and ( \
                                                 (possible_polygon['geometry'].contains(polygon['geometry'])) | \
                                                 (possible_polygon['geometry'].overlaps(polygon['geometry']))   \
                                                )  :  
                    
                    if possible_polygon['geometry'].area >= polygon['geometry'].area: 
                        if index not in to_remove : 
                            to_remove.append(index)
                            flag_rm = True
                            break

            if False: # 5700 in possible_index_arr: 
                ax = plt.subplot(111)
                ax.set_title('rm = {:b}, 1 is True'.format(flag_rm))
                possible_matches.plot(alpha=0.3,ax=ax)
                merged[index:index+1].plot(color='none',edgecolor='r',ax=ax)
                plt.show()

                pdb.set_trace()

        # Remove polygons that are contained within others
        merged_filtered = merged.drop(index=to_remove)


        #save
        merged_filtered.to_file('{:s}/tourism-{:s}.geojson'.format(dirout,name), driver='GeoJSON')

   

    sys.exit()






    #set tourism_pt on settlement map
    row, col = pts2index(tourism_pt.geometry.x, tourism_pt.geometry.y)
    row, col = np.array(row), np.array(col)
    settlement_mask_tourism_pt = np.zeros_like(settlement)
    idx = np.where( (row>0)&(row<settlement.shape[0])&(col>0)&(col<settlement.shape[1]) )
    settlement_mask_tourism_pt[row[idx], col[idx]] = 1

    dx=1000
    dy=1000
    xkde = np.arange(0,Lx,dx) + settlement_origin_x
    ykde = np.arange(0,Ly,dy) + settlement_origin_y
    
    yykde, xxkde = np.meshgrid(ykde,xkde)

    values = np.vstack([tourism_pt.geometry.x, tourism_pt.geometry.y])
    kernel = stats.gaussian_kde(values)
    
    positions = np.vstack([xxkde.ravel(), yykde.ravel()])
    f = np.reshape(kernel(positions).T, xxkde.shape)

   
    sys.exit()



    #keep settlement with at leat 3 tourism point within 500m
    mask_tourism = np.zeros_like(settlement)
    #labels, nlabels = ndimage.label(np.where(settlement==255,1,0), structure=np.ones([3,3]))

    mwx, mwy = 1000,1000
    nx,ny = settlement.shape

    for i in range(mwx//2,nx-mwx//2)[::200]:
        for j in range(mwy//2,ny-mwy//2)[::200]:

            print(i,j,nx,ny)
            settlement_ = settlement[i-mwx//2:i+mwx//2,j-mwy//2:j+mwy//2]
            labels_, nlabels_ = ndimage.label(np.where(settlement_==255,1,0), structure=np.ones([3,3]))


            for ilabel in range(nlabels_):
                #print( ilabel,  nlabels_, end='')
                sys.stdout.flush()
                idx = np.where(labels_ == ilabel+1)
                area = idx[0].shape[0] * dx**2
                
                if area < 100: 
                    #print(' <')
                    continue 
                
                #create local mask
                mask_ = np.zeros_like(settlement_)
                mask_[idx] = 1
                #mask_ = cv2.morphologyEx(mask_, cv2.MORPH_CLOSE, np.ones([3,3]))

                #add buffer
                mask_buffered = cv2.dilate(mask_,np.ones([3,3]),iterations = int(500./dx))

                if settlement_mask_tourism_pt[np.where( mask_buffered  == 1)].sum() > 3:
                    idx_full =( idx[0]+(i-mwx/2), idx[1]+(j-mwy/2) )
                    mask_tourism[idx_full] = 1
                    #print(' *')
                    sys.exit()
                #else:
                #    print('')

    ax = plt.subplot(111)
    show(settlement, transform=settlement_transform, ax=ax) 
    tourism_pt.plot(ax=ax, color='k')
    plt.show()


