import os, argparse
# import geopandas as gpd
import pandas as pd
import gdal
import numpy as np
import geosardine as dine
import fiona
import rasterio
import rasterio.mask



class Renderer(object):
    def __init__(self, map_shp, crs=4326):
        # save attrs
        self.crs = crs
        # open map as shapes
        with fiona.open(map_shp, "r") as shapefile:
            self.shapes = [feature["geometry"] for feature in shapefile]


    def render(self, df, vars, out_dir='.', spatial_res=(0.1,0.1), lon_key = 'lon', lat_key = 'lat'):
        # render each var
        for cur_var in vars:
            print('Rendering variable: {}'.format(cur_var))
            # interpolate points using IDW
            interpolated = dine.interpolate.idw(
                                        df[[lon_key, lat_key]].to_numpy(),
                                        df[cur_var].to_numpy(),
                                        spatial_res=spatial_res, epsg=self.crs)
            # get vars
            xmin,xmax,ymin,ymax = interpolated.extent
            west, north = xmin, ymax
            nrows, ncols = interpolated.array.shape
            # create transform
            transform = rasterio.transform.from_origin(west, north, spatial_res[0], spatial_res[1])
            # create output filename
            cur_out_fn = os.path.join(out_dir, cur_var + '.tif')
            # define array
            arr = interpolated.array
            # save output in geotif
            new_dataset = rasterio.open(cur_out_fn, 'w', driver='GTiff',
                                        height = arr.shape[0], width = arr.shape[1],
                                        count=1, dtype=str(arr.dtype),
                                        crs=4326,
                                        transform=transform)
            new_dataset.write(arr, 1)
            new_dataset.close()
            # mask with shapes
            with rasterio.open(cur_out_fn) as src:
                out_image, out_transform = rasterio.mask.mask(src, self.shapes, crop=True, all_touched=True, nodata=0, filled=True)
                out_meta = src.meta
            # update meta
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "nodata":0})
            # save result
            with rasterio.open(cur_out_fn, "w", **out_meta) as dest:
                dest.write(out_image)




if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    # checkpoint path
    parser.add_argument("-csv", "--csv", help="csv file.",
    					default='', type=str)
    parser.add_argument("-sep", "--sep", help="csv file column separator.",
    					default=';', type=str)
    parser.add_argument("-vars", "--vars", help="list of vars.", nargs='+',
    					default=['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'], type=str)
    parser.add_argument("-lat", "--lat", help="name of the latitude field.",
    					default='lat', type=str)
    parser.add_argument("-lon", "--lon", help="name of the longitude field.",
    					default='lon', type=str)
    parser.add_argument("-map", "--map", help="Shapefile of the map.",
    					default='', type=str)
    parser.add_argument("-crs", "--crs", help="CRS code.",
    					default=4326, type=int)
    parser.add_argument("-sr", "--spatial_resolution", help="spatial_resolution.", nargs=2,
    					default=(0.1, 0.1), type=float)
    parser.add_argument("-outdir", "--outdir", help="Output directory.",
    					default='', type=str)
    # parse args
    args = parser.parse_args()
    # if outdir not specified, use same as csv file
    if args.outdir == '':
        args.outdir = os.path.dirname(args.csv)
    # load dataframe
    df = pd.read_csv(args.csv, sep=args.sep)
    # filter vars that are not included
    correct_cols = [cur_var for cur_var in args.vars if cur_var in df.columns]
    # init renderer
    renderer = Renderer(args.map)
    # render all maps
    renderer.render(df, correct_cols, out_dir=args.outdir, spatial_res=args.spatial_resolution, lon_key=args.lon, lat_key=args.lat)
    
    # https://github.com/mapbox/rasterio/issues/1178
