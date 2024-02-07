import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from shapely.validation import make_valid, explain_validity
import warnings


if __name__ == '__main__':

    dir_data = '/mnt/dataMoor/WUITIPS/'

    filesin  = [dir_data+'WUI/WUIall.geojson', dir_data+'TourismSpots/spotsall_withflag.geojson' ]
    filesout = [dir_data+'Mapbox/WUI.geojson',          dir_data+ 'Mapbox/spots.geojson'                  ]         


    for filein, fileout in zip(filesin, filesout):
       
        print(filein)
        gdf = gpd.read_file(filein)

        #convert to wgs
        gdf = gdf.to_crs('epsg:4326')


        #valide geom
        gdf.geometry = gdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)
        
        for index, row in gdf[(gdf.geom_type != 'Polygon') & (gdf.geom_type!='MultiPolygon')].iterrows():
            with warnings.catch_warnings(record=True) as w:
                gdf.at[index,'geometry'] =  gdf[index:index+1].geometry.buffer(1.e-10).unary_union 

        gdf = gdf[~gdf['geometry'].is_empty]

        #save
        gdf.to_file( fileout )

