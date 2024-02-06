#!/bin/bash
export MAPBOX_ACCESS_TOKEN=sk.eyJ1Ijoicm9uYW4tcDMzIiwiYSI6ImNsamN3NWEzeDB6Nm0zZXFocHM5c2VxN2QifQ.GCF3TvmWUi-LK5KESrm5qg
source ~/anaconda3/bin/activate mapbox
#ogr2ogr -f GeoJSONSeq ./Mapbox/spots.geojson.ld ./Mapbox/spots.geojson
tilesets upload-source ronan-p33 wuitips-spots ./Mapbox/spots.geojson
tilesets create ronan-p33.wuitips-spots --recipe ./Mapbox/recipe-wuitips-spots.json --name "wuitips spots"
tilesets publish ronan-p33.wuitips-spots

#ogr2ogr -f GeoJSONSeq ./Mapbox/WUI.geojson.ld ./Mapbox/WUI.geojson
tilesets upload-source ronan-p33 wuitips-WUI ./Mapbox/WUI.geojson
tilesets create ronan-p33.wuitips-WUI --recipe ./Mapbox/recipe-wuitips-WUI.json --name "wuitips WUI"
tilesets publish ronan-p33.wuitips-WUI

