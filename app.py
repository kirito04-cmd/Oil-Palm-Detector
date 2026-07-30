import streamlit as st
import os
import cv2
import numpy as np
import rasterio
import rasterio.windows
import geopandas as gpd
import zipfile
import tempfile
import shutil
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
        indices = tree.query_ball_point(point, r=distance_threshold_meters)
        visited.update(indices)
        
    return keep

# Safely render overlay visualization
def render_detection_preview(tif_path, gdf, max_dim=1024):
    with rasterio.open(tif_path) as src:
        orig_w, orig_h = src.width, src.height
        scale = max_dim / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        # Read downsampled RGB bands
        data = src.read([1, 2, 3], out_shape=(3, new_h, new_w))
        data = np.moveaxis(data, 0, -1)

        if data.dtype != np.uint8:
            data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(data)

        if not gdf.empty:
            pixel_x = []
            pixel_y = []
            for point in gdf.geometry:
                row, col = src.index(point.x, point.y)
                pixel_x.append(col * scale)
                pixel_y.append(row * scale)
            ax.scatter(pixel_x, pixel_y, c='red', s=15, marker='o', label='Detected Trees')
            ax.legend(loc="upper right")

        ax.axis('off')
        plt.tight_layout()
        
        return fig

# UI Layout
st.title("🌴 Large-Scale Oil Palm Plantation Detector")
st.write("Process full orthomosaics (up to 10GB+) using tiled memory management and automatic coordinate deduplication.")

# Sidebar Settings
st.sidebar.header("AI Settings")
api_key = st.sidebar.text_input("Roboflow API Key", value="yVaMpDjeXPH2Mzqs41u7", type="password")
confidence_setting = st.sidebar.slider("Confidence Limit (%)", min_value=1, max_value=100, value=60)
overlap_setting = st.sidebar.slider("AI Internal Overlap (%)", min_value=1, max_value=100, value=25)

st.sidebar.header("Tiling Settings")
tile_size = st.sidebar.selectbox("Tile Size (Pixels)", [512, 640, 1024], index=2)
tile_overlap = st.sidebar.slider("Tile Overlap (Pixels)", min_value=32, max_value=256, value=128, step=32)
dedup_distance = st.sidebar.slider("Duplicate Merge Distance (Meters)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

# Input method option for large files
input_method = st.radio("File Source:", ["Upload File (Small/Medium)", "Local File Path (Best for 5GB-10GB Files)"])

tif_path = None
temp_uploaded_file = None

if input_method == "Upload File (Small/Medium)":
    uploaded_file = st.file_uploader("Upload GeoTIFF (.tif)", type=["tif", "tiff"])
    if uploaded_file is not None:
        temp_uploaded_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        while chunk := uploaded_file.read(8 * 1024 * 1024):
            temp_uploaded_file.write(chunk)
        temp_uploaded_file.close()
        tif_path = temp_uploaded_file.name
else:
    tif_path = st.text_input("Enter absolute file path on server/computer (e.g. /content/large_plot.tif or C:/data/plot.tif)")

# Store raw prediction objects in session state to allow instant filtering
if tif_path and os.path.exists(tif_path):
    if st.button("🚀 Run Full Plot Detection"):
        temp_dir = tempfile.mkdtemp()
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
                
                raw_predictions = []  # Holds tuples of (Point, confidence)
                tile_count = 0
                temp_tile_path = os.path.join(temp_dir, "tile.jpg")

                # Run tile loop at minimum threshold (1%) to capture all candidates once
                for y in y_steps:
                    for x in x_steps:
                        tile_count += 1
                        
                        w_width = min(tile_size, width - x)
                        w_height = min(tile_size, height - y)
                        window = rasterio.windows.Window(x, y, w_width, w_height)

                        tile_data = src.read(window=window)
                        if tile_data.shape[0] >= 3:
                            tile_data = tile_data[:3, :, :]
                        tile_data = np.moveaxis(tile_data, 0, -1)

                        if tile_data.dtype != np.uint8:
                            tile_data = cv2.normalize(tile_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        if np.mean(tile_data) < 5:
                            continue

                        cv2.imwrite(temp_tile_path, cv2.cvtColor(tile_data, cv2.COLOR_RGB2BGR))

                        # Query model with 1% confidence to get full candidate list
                        preds = model.predict(temp_tile_path, confidence=1, overlap=overlap_setting).json()

                        if "predictions" in preds:
                            for p in preds["predictions"]:
                                global_pixel_x = x + p["x"]
                                global_pixel_y = y + p["y"]

                                map_x, map_y = transform * (global_pixel_x, global_pixel_y)
                                conf = p.get("confidence", 1.0)
                                if conf <= 1.0:
                                    conf = conf * 100  # Normalize to 0-100 scale if needed

                                raw_predictions.append({"point": Point(map_x, map_y), "confidence": conf})

                        progress_percent = int((tile_count / total_tiles) * 100)
                        progress_bar.progress(
                            progress_percent, 
                            text=f"Scanning Tile {tile_count}/{total_tiles} ({progress_percent}%) | Candidates: {len(raw_predictions)}"
                        )

                progress_bar.empty()

                # Save raw candidate cache into Session State
                st.session_state['cached_predictions'] = raw_predictions
                st.session_state['crs'] = crs
                st.session_state['tif_path'] = tif_path

        except Exception as e:
            st.error(f"Error while running plot detection: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Process and Render Results instantly when confidence slider moves
    if 'cached_predictions' in st.session_state and st.session_state.get('tif_path') == tif_path:
        raw_preds = st.session_state['cached_predictions']
        crs = st.session_state['crs']

        # Filter points by current confidence slider
        filtered_points = [
            item["point"] for item in raw_preds 
            if item["confidence"] >= confidence_setting
        ]

        # Apply spatial deduplication
        final_points = filter_duplicate_points(filtered_points, distance_threshold_meters=dedup_distance)

        st.success(f"🎉 Displaying Detections: {len(final_points)} Palms at {confidence_setting}% Confidence Threshold")

        gdf = gpd.GeoDataFrame(geometry=final_points, crs=crs)

        if len(final_points) > 0:
            if crs is not None:
                gdf_wgs84 = gdf.to_crs(epsg=4326)
                gdf['Latitude'] = gdf_wgs84.geometry.y.astype(float)
                gdf['Longitude'] = gdf_wgs84.geometry.x.astype(float)
            else:
                gdf['Latitude'] = gdf.geometry.y.astype(float)
                gdf['Longitude'] = gdf.geometry.x.astype(float)
                
            gdf['Altitude'] = 0.0

            col1, col2 = st.columns([1, 2])
            with col2:
                st.write("Attributes Preview:")
                st.dataframe(gdf[['Latitude', 'Longitude', 'Altitude']].head(10))

        # Render Image Detection Preview
        st.markdown("---")
        st.subheader("🖼️ Detection Visualizer")
        fig_preview = render_detection_preview(tif_path, gdf)
        st.pyplot(fig_preview)
        plt.close(fig_preview)
