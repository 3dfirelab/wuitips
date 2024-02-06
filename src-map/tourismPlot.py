import geopandas as gpd
from shapely.validation import make_valid, explain_validity
import warnings
import matplotlib.pyplot as plt
import contextily 

if __name__ == '__main__':
    
    WUIFile = './Mapbox/WUI.geojson'         
    spotFile = './Mapbox/spots.geojson'

    gdf_spot = gpd.read_file(spotFile)
    gdf_wui = gpd.read_file(WUIFile)

    gdf_spot = gdf_spot.to_crs('epsg:3035')
    gdf_wui = gdf_wui.to_crs('epsg:3035')

    ax = plt.subplot(111)
    gdf_wui.plot(ax=ax, alpha=.5, color='r')
    gdf_spot.plot(ax=ax, alpha=.5, color='c')

    w,s,e,n = gdf_spot.total_bounds
    
    contextily.add_basemap(ax, crs=gdf_spot.crs, source=contextily.providers.OpenStreetMap.Mapnik, zoom=10)

    plt.show()



