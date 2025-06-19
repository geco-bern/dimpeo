import pystac_client
import rasterio
from rasterio.coords import BoundingBox
import numpy as np
import h5py
from tqdm import tqdm
from rasterio.windows import from_bounds

# Connect to Swisstopo STAC API
service = pystac_client.Client.open('https://data.geo.admin.ch/api/stac/v0.9/')
service.add_conforms_to("COLLECTIONS")
service.add_conforms_to("ITEM_SEARCH")

# EPSG: 4326
# WGS 84
# Swiss bounds: left, bottom, right, top
bbox_swiss_4326 = [5.70, 45.8, 10.6, 47.95]

# Retrieve the spatial coverage (bounds) of all 4 possible orbits covering Switzerland
def collect_bounds_all_orbits():
    """
    Collects the bounds of all orbits in the Swiss dataset.
    Returns a list of BoundingBox objects.
    """
    item_search = service.search(
        bbox=bbox_swiss_4326,
        datetime='2025-04-30/2025-05-02',
        collections=['ch.swisstopo.swisseo_s2-sr_v100']
    )
    s2_files_sample_orbits = list(item_search.items())

    all_bounds = []

    for item in tqdm(s2_files_sample_orbits):
        assets = item.assets
        key_bands = [k for k in assets.keys() if k.endswith('bands-10m.tif')][0]
        bands_asset = assets[key_bands]
        with rasterio.open(bands_asset.href) as src:
            bounds = src.bounds
            all_bounds.append(bounds)

    return all_bounds

# Combine all bounding boxes into one global bounding box and compute its pixel dimensions
def union_bounds(bounds_list):
    """
    Takes a list of BoundingBox objects and returns a single BoundingBox
    that encompasses all the bounds, along with the width and height
    of the bounding box in pixels, assuming a resolution of 10 meters.
    """
    left = min(b.left for b in bounds_list)
    bottom = min(b.bottom for b in bounds_list)
    right = max(b.right for b in bounds_list)
    top = max(b.top for b in bounds_list)
    resolution = 10
    width = int((right - left) / resolution)
    height = int((top - bottom) / resolution)
    return BoundingBox(left, bottom, right, top), width, height

all_bounds = collect_bounds_all_orbits()

# EPSG: 2056
# Swiss coordinate system (CH1903+ / LV95)
# This is the full reference bounding box for the Swisstopo dataset covering the 4 orbits
bbox_swisstopo_2056, width_swisstopo, height_swisstopo = union_bounds(all_bounds)

# Derive the forest mask from the Swisstopo VHI dataset 
# The VHI dataset contains the forest mask that Swisstopo derived from the habitat map
# Also collect the metadata using the forest mask as a reference raster
def get_forest_mask():
    """
    Downloads the forest mask from the Swisstopo VHI dataset.
    Returns a numpy array representing the forest mask.
    Also returns the metadata for the reference raster.
    """
    item_search = service.search(
        bbox=bbox_swiss_4326,
        datetime='2025-05-01/2025-05-01',
        collections=['ch.swisstopo.swisseo_vhi_v100']
    )
    items = list(item_search.items())
    item = items[0]
    assets = item.assets
    key_bands = [k for k in assets.keys() if k.endswith('forest-10m.tif')][0]
    bands_asset = assets[key_bands]
    
    with rasterio.open(bands_asset.href) as src:
        window = src.window(*bbox_swisstopo_2056)
        vhi = src.read(1, window=window)
        forest_mask = (vhi != 255).astype('uint8')
        ref_meta = {
            "transform": src.window_transform(window),
            "crs": src.crs,
            "width": window.width,
            "height": window.height
        }
    
    return forest_mask, ref_meta

forest_mask, ref_meta = get_forest_mask()
print("Reference raster metadata:")
print(ref_meta)

# Build index mapping from forest pixels in the full reference raster to 1D flat indices
forest_flat_indices = np.flatnonzero(forest_mask == 1)
max_index = forest_flat_indices.max() + 1
index_map = np.full(max_index, -1, dtype=np.int32)
index_map[forest_flat_indices] = np.arange(len(forest_flat_indices))

# Search all images for the full CH bounding box for the whole time period
item_search = service.search(
    bbox=bbox_swiss_4326,
    datetime='2017-04-01/2025-05-31',
    collections=['ch.swisstopo.swisseo_s2-sr_v100']
)
s2_files = list(item_search.items())

# Prepare constants
N = len(forest_flat_indices)
T = len(s2_files)
NDVI_INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows
NDVI_NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step

# Create HDF5 file to store the NDVI time series for forest pixels
with h5py.File("/data_2/scratch/sbiegel/processed/new_ndvi_timeseries.h5", "w") as h5f:
    # Define the dataset for NDVI values
    # Shape is (T, N) where T is the number of time steps and N is the number of forest pixels
    # Use int16 to save space, with a fill value for no coverage
    # Use compression to save space
    # Use chunks to allow for efficient reading and writing
    ndvi_ds = h5f.create_dataset(
        "ndvi",
        shape=(T, N),
        maxshape=(T, N),
        dtype="int16",
        chunks=(1, N),
        compression="lzf",
        fillvalue=NDVI_NO_COVERAGE
    )

    for t, ndvi_path in tqdm(enumerate(s2_files), total=len(s2_files)):
        # Get the assets for the current time step
        assets = ndvi_path.assets
        bands_asset = assets[[k for k in assets.keys() if k.endswith('bands-10m.tif')][0]]
        masks_asset = assets[[k for k in assets.keys() if k.endswith('masks-10m.tif')][0]]

        # Read the NDVI bands and masks
        with rasterio.open(bands_asset.href) as bands_src, rasterio.open(masks_asset.href) as masks_src:

            if not ((bands_src.transform == masks_src.transform) and (bands_src.width, bands_src.height) == (masks_src.width, masks_src.height)):
                # Handle the case where masks and bands are not aligned
                print("Warning: Transforms or dimensions do not match between bands and masks assets for time step", t)
                # If the transforms or bounds do not match, we need to sample the entire window
                # to ensure we get the same area for both bands and masks.
                band_window = from_bounds(*bbox_swisstopo_2056, transform=bands_src.transform)
                mask_window = from_bounds(*bbox_swisstopo_2056, transform=masks_src.transform)
                red, nir = bands_src.read([1, 4], window=band_window, boundless=True, fill_value=9999)
                masks = masks_src.read([1, 2], window=mask_window, boundless=True, fill_value=255).astype("uint8")
                terrain_mask, cloud_mask = masks
            else:
                # Standard case: masks and bands are aligned
                window = bands_src.window(*bbox_swisstopo_2056)
                red, nir = bands_src.read([1, 4], window=window)
                masks = masks_src.read([1, 2], window=window).astype("uint8")
                terrain_mask, cloud_mask = masks

        # Create masks for cloud shadows and nodata
        cloud_shadows_mask = (terrain_mask == 100) | (cloud_mask == 1)
        nodata_mask = (red == 9999) | (nir == 9999) | (terrain_mask == 255) | (cloud_mask == 255)
        
        # Calculate NDVI
        red = red.astype("float32") / 10000.0
        nir = nir.astype("float32") / 10000.0
        ndvi = (nir - red) / (nir + red)
        ndvi = np.clip(ndvi, -1.0, 1.0)
        ndvi_scaled = (ndvi * 10000.0).astype("int16")

        # Compute window offset in the reference raster grid (forest mask)
        # This is the area of the reference raster that corresponds to the current bands raster
        window = from_bounds(*bands_src.bounds, transform=ref_meta["transform"]).round_offsets().round_lengths()
        row_start, row_stop = window.row_off, window.row_off + window.height
        col_start, col_stop = window.col_off, window.col_off + window.width

        #  Extract forest pixels from local window (current subraster)
        local_forest_mask = forest_mask[row_start:row_stop, col_start:col_stop]
        local_rows, local_cols = np.where(local_forest_mask)

        # Map local rows and columns to global indices in the full reference raster
        global_rows = local_rows + row_start
        global_cols = local_cols + col_start
        global_flat = global_rows * width_swisstopo + global_cols
        # Get the corresponding flat indices (column indices) in the index map
        current_flat_indices = index_map[global_flat]

        # Get flat NDVI and masks for the current time step
        ndvi_flat = ndvi_scaled[local_rows, local_cols]
        cloud_shadows_mask_flat = cloud_shadows_mask[local_rows, local_cols]
        nodata_mask_flat = nodata_mask[local_rows, local_cols]

        # Prepare the NDVI row for the current time step
        ndvi_row = np.full(N, NDVI_NO_COVERAGE, dtype="int16")
        valid = ~(cloud_shadows_mask_flat | nodata_mask_flat)
        cloud_only = cloud_shadows_mask_flat & ~nodata_mask_flat
        ndvi_row[current_flat_indices[valid]] = ndvi_flat[valid]
        ndvi_row[current_flat_indices[cloud_only]] = NDVI_INVALID
        ndvi_ds[t] = ndvi_row
