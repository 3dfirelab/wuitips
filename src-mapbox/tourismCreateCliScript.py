import json 
import sys
import importlib 
import glob
import os

if __name__ == '__main__':

    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('export MAPBOX_ACCESS_TOKEN=sk.{:s}\n'.format(os.environ['MAPBOX_TOKEN']))
    lines.append('source ~/anaconda3/bin/activate mapbox')
    lines.append('\n')

    dir_data = '/mnt/dataMoor/WUITIPS/'
    indir = dir_data+'Mapbox/' 
    for file_ in glob.glob(indir+'*.geojson'):
    
        fileValide   = file_
        fileValideld = '{:s}.ld'.format(file_)

        name = os.path.basename(file_).split('.')[0]+'v3'

        recipe = {}
        recipe['version'] = 1
        recipe['layers']  = {
                             'wuitips-{:s}'.format(name):{
                                                           'source':"mapbox://tileset-source/ronan-p33/wuitips-{:s}".format(name), 
                                                           'minzoom': 0,
                                                           'maxzoom': 15,
                                                           } 
                             }
       
        with open('{:s}recipe-wuitips-{:s}.json'.format(indir,name), 'w') as fp:
            json.dump(recipe, fp)

        #lines.append('ogr2ogr -f GeoJSONSeq {:s} {:s}\n'.format(fileValideld, fileValide) )
        #lines.append('tilesets upload-source ronan-p33 wuitips-{:s} {:s}\n'.format(name, fileValideld))
        lines.append('tilesets upload-source ronan-p33 wuitips-{:s} {:s}\n'.format(name, fileValide))
        lines.append('tilesets create ronan-p33.wuitips-{:s} --recipe {:s}recipe-wuitips-{:s}.json --name "wuitips {:s}"\n'.format(name,indir,name, name))
        lines.append('tilesets publish ronan-p33.wuitips-{:s}\n'.format(name))

        lines.append('\n')
 
    with open('run_push2Mapbox.sh', 'w') as fp:
        fp.writelines(lines)
    
