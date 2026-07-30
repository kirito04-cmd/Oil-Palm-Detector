import streamlit as st
import os
import cv2
import numpy as np
import rasterio
import rasterio.windows
import geopandas as gpd
import zipfile
import tempfile
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.spatial import cKDTree
from roboflow import Roboflow

st.set_page_config(page_title="Oil Palm Detection", layout="wide")

# Cache model connection
@st.cache_resource
def load_roboflow_model(api_key, version_number):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("hanifs-workspace-bd93u").project("oil-palm-tree-detection-sv9gl")
    return project.version(version_number).model

# Distance-based NMS to remove duplicate detections from overlapping tiles
def filter_duplicate_points(map_points, distance_threshold_meters=2.0):
    if not map_points:
        return []
    
    coords = np.array([[p.x, p.y] for p in map_points])
    tree = cKDTree(coords)
    
    keep = []
    visited = set()
    
    for i, point in enumerate(coords):
        if i in visited:
            continue
        keep.append(map_points[i])
        # Find all points within the threshold distance and mark them as duplicate
        indices = tree.query_ball_point(point, r=distance_threshold_meters)
        visited.update(indices)
        
    return keep

st.title("🌴 Large-Scale Oil Palm Plantation Detector")
st.write("Process full orthomosaics (up to 10GB+) using tiled memory management and automatic coordinate deduplication.")

# Sidebar Settings
st.sidebar.header("AI Settings")
api_key = st.sidebar.text_input("Roboflow API Key", value="yVaMpDjeXPH2Mzqs41u7", type="password")
confidence_setting = st.sidebar.slider("Confidence Limit (%)", min_value=1, max_value=100, value=15)
overlap_setting = st.sidebar.slider("AI Internal Overlap (%)", min_value=1, max_value=100, value=30)

st.sidebar.header("Tiling Settings")
tile_size = st.sidebar.selectbox("Tile Size (Pixels)", [512, 640, 1024], index=2)
tile_overlap = st.sidebar.slider("Tile Overlap (Pixels)", min_value=32, max_value=256, value=128, step=32)
dedup_distance = st.sidebar.slider("Duplicate Merge Distance (Meters)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

# Input method option for large files
input_method = st.radio("File Source:", ["Upload File (Small/Medium)", "Local File Path (Best for 5GB-10GB Files)"])

tif_path = None

if input_method == "Upload File (Small/Medium)":
    uploaded_file = st.file_uploader("Upload GeoTIFF (.tif)", type=["tif", "tiff"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_tif:
            temp_tif.write(uploaded_file.read())
            tif_path = temp_tif.name
else:
    tif_path = st.text_input("Enter absolute file path on server/computer (e.g. /content/large_plot.tif or C:/data/plot.tif)")

if tif_path and os.path.exists(tif_path):
    if st.button("🚀 Run Full Plot Detection"):
        try:
            model = load_roboflow_model(api_key, 10)
            
            with rasterio.open(tif_path) as src:
                width = src.width
                height = src.height
                transform = src.transform
                crs = src.crs

                stride = tile_size - tile_overlap
                x_steps = list(range(0, width, stride))
                y_steps = list(range(0, height, stride))
                total_tiles = len(x_steps) * len(y_steps)

                st.info(f"📊 Image Dimensions: {width}x{height} px | Grid Split: {total_tiles} Tiles")

                progress_bar = st.progress(0, text="Starting grid tile scan...")
                
                raw_map_points = []
                tile_count = 0

                temp_dir = tempfile.mkdtemp()
                temp_tile_path = os.path.join(temp_dir, "tile.jpg")

                # Iterate through grid windows without loading full image to RAM
                for y in y_steps:
                    for x in x_steps:
                        tile_count += 1
                        
                        # Define pixel window
                        w_width = min(tile_size, width - x)
                        w_height = min(tile_size, height - y)
                        window = rasterio.windows.Window(x, y, w_width, w_height)

                        # Read only current tile window
                        tile_data = src.read(window=window)
                        if tile_data.shape[0] >= 3:
                            tile_data = tile_data[:3, :, :]
                        tile_data = np.moveaxis(tile_data, 0, -1)

                        if tile_data.dtype != np.uint8:
                            tile_data = cv2.normalize(tile_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        # Skip blank/black background tiles
                        if np.mean(tile_data) < 5:
                            continue

                        cv2.imwrite(temp_tile_path, cv2.cvtColor(tile_data, cv2.COLOR_RGB2BGR))

                        # Predict on single tile
                        preds = model.predict(temp_tile_path, confidence=confidence_setting, overlap=overlap_setting).json()

                        if "predictions" in preds:
                            for p in preds["predictions"]:
                                # Convert tile-relative coordinates to full image coordinates
                                global_pixel_x = x + p["x"]
                                global_pixel_y = y + p["y"]

                                # Translate to map GIS coordinates
                                map_x, map_y = transform * (global_pixel_x, global_pixel_y)
                                raw_map_points.append(Point(map_x, map_y))

                        # Progress update
                        progress_percent = int((tile_count / total_tiles) * 100)
                        progress_bar.progress(
                            progress_percent, 
                            text=f"Scanning Tile {tile_count}/{total_tiles} ({progress_percent}%) | Detected: {len(raw_map_points)} Palms"
                        )

                progress_bar.progress(100, text="🧹 Removing duplicate boundary detections...")
                
                # Filter boundary duplicates
                final_points = filter_duplicate_points(raw_map_points, distance_threshold_meters=dedup_distance)
                progress_bar.empty()

                st.success(f"🎉 Plot Scanning Complete! Total Trees Detected: {len(final_points)} (Removed {len(raw_map_points) - len(final_points)} duplicates)")

                # Export Shapefile
                if len(final_points) > 0:
                    gdf = gpd.GeoDataFrame(geometry=final_points, crs=crs)
                    
                    if crs is not None:
                        gdf_wgs84 = gdf.to_crs(epsg=4326)
                        gdf['Latitude'] = gdf_wgs84.geometry.y.astype(float)
                        gdf['Longitude'] = gdf_wgs84.geometry.x.astype(float)
                    else:
                        gdf['Latitude'] = gdf.geometry.y.astype(float)
                        gdf['Longitude'] = gdf.geometry.x.astype(float)
                        
                    gdf['Altitude'] = 0.0

                    output_base = os.path.join(temp_dir, "detected_palm_centers")
                    gdf.to_file(output_base + '.shp', driver="ESRI Shapefile")

                    zip_path = os.path.join(temp_dir, "detected_palm_centers.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                            file_part = output_base + ext
                            if os.path.exists(file_part):
                                zipf.write(file_part, os.path.basename(file_part))

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="💾 Download Plot Shapefile (.zip)",
                                data=f,
                                file_name="detected_palm_centers.zip",
                                mime="application/zip"
                            )
                    with col2:
                        st.write("Attributes Preview:")
                        st.dataframe(gdf[['Latitude', 'Longitude', 'Altitude']].head(10))

        except Exception as e:
            st.error(f"Error while running plot detection: {e}")
