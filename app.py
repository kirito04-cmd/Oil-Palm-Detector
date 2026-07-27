import streamlit as st
import os
import cv2
import numpy as np
import rasterio
import geopandas as gpd
import zipfile
import tempfile
import time
from shapely.geometry import Point
from roboflow import Roboflow

st.set_page_config(page_title="Oil Palm Tree Detector", layout="wide")

# =====================================================================
# CACHED MODEL LOADER
# =====================================================================
@st.cache_resource
def load_roboflow_model(api_key, version_number):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("hanifs-workspace-bd93u").project("oil-palm-tree-detection-sv9gl")
    return project.version(version_number).model

st.title("🌴 Live Oil Palm Tree Detection")
st.write("Upload your drone orthophoto (.tif) to watch the AI detect tree crowns live, one by one.")

# Sidebar Settings
st.sidebar.header("AI Settings")
api_key = st.sidebar.text_input("Roboflow API Key", value="yVaMpDjeXPH2Mzqs41u7", type="password")
confidence_setting = st.sidebar.slider("Confidence Limit (%)", min_value=1, max_value=100, value=5)
overlap_setting = st.sidebar.slider("Overlap Limit (%)", min_value=1, max_value=100, value=50)

st.sidebar.header("Live Visual Controls")
# Controls how fast each tree dot appears on screen
anim_delay = st.sidebar.slider("Live Speed Delay (sec)", min_value=0.01, max_value=0.50, value=0.08, step=0.01)

# File Uploader
uploaded_file = st.file_uploader("Upload Drone GeoTIFF Image (.tif)", type=["tif", "tiff"])

if uploaded_file is not None:
    # -----------------------------------------------------------------
    # STEP 1: INITIAL LOADING BARS
    # -----------------------------------------------------------------
    load_progress = st.progress(0, text="📂 Uploading and opening image file...")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_tif:
        temp_tif.write(uploaded_file.read())
        temp_tif_path = temp_tif.name

    load_progress.progress(25, text="🔄 Reading spatial coordinates from GeoTIFF metadata...")

    with rasterio.open(temp_tif_path) as src:
        transform = src.transform
        crs = src.crs
        img_data = src.read()
        if len(img_data.shape) == 3:
            img_data = np.moveaxis(img_data, 0, -1)

    load_progress.progress(50, text="🖼️ Normalizing RGB image channels...")

    if img_data.dtype != np.uint8:
        img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if len(img_data.shape) == 2 or img_data.shape[-1] == 1:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
    elif img_data.shape[-1] > 3:
        img_data = img_data[:, :, :3]

    temp_jpg_path = os.path.join(tempfile.gettempdir(), "temp_ready.jpg")
    cv2.imwrite(temp_jpg_path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))

    load_progress.progress(75, text="🧠 Requesting YOLOv11 predictions from Roboflow...")

    try:
        model = load_roboflow_model(api_key=api_key, version_number=10)
        predictions = model.predict(temp_jpg_path, confidence=confidence_setting, overlap=overlap_setting).json()
        
        load_progress.progress(100, text="✅ AI Model response received! Starting live detection...")
        time.sleep(0.5)
        load_progress.empty()  # Clear initial loading bar

        # -----------------------------------------------------------------
        # STEP 2: LIVE ONE-BY-ONE DETECTION FEED
        # -----------------------------------------------------------------
        st.subheader("📡 Live AI Tree Crown Detection Feed")
        
        # Placeholders for live visual updates
        live_status = st.empty()
        detection_progress = st.progress(0)
        live_image_display = st.empty()

        raw_preds = predictions.get("predictions", [])
        total_found = len(raw_preds)

        map_points = []
        # Create an working copy of the image array to draw green dots on
        display_canvas = img_data.copy()

        if total_found > 0:
            for idx, pred in enumerate(raw_preds):
                pixel_x = int(pred["x"])
                pixel_y = int(pred["y"])

                # Calculate GIS spatial coordinates
                map_x, map_y = transform * (pixel_x, pixel_y)
                map_points.append(Point(map_x, map_y))

                # Draw outer black border circle and inner green target dot
                cv2.circle(display_canvas, (pixel_x, pixel_y), 10, (0, 0, 0), -1)
                cv2.circle(display_canvas, (pixel_x, pixel_y), 7, (0, 255, 0), -1)

                # Update live image display
                live_image_display.image(display_canvas, caption=f"Live Feed: {idx + 1} / {total_found} Palm Crowns Found", use_container_width=True)
                
                # Update status message and progress bar
                live_status.markdown(f"🎯 **Detecting Tree #{idx + 1}** at Pixel `({pixel_x}, {pixel_y})` | GIS Map: `({map_x:.2f}, {map_y:.2f})`")
                detection_progress.progress((idx + 1) / total_found)

                # Delay to create the step-by-step visual effect
                time.sleep(anim_delay)

            st.success(f"🎉 Detection finished! Mapped all {total_found} tree crowns successfully.")

            # -------------------------------------------------------------
            # STEP 3: EXPORT SHAPEFILE & ATTRIBUTES
            # -------------------------------------------------------------
            st.divider()
            st.subheader("💾 Export GIS Shapefile")
            
            gdf = gpd.GeoDataFrame(geometry=map_points, crs=crs)
            
            if crs is not None:
                gdf_wgs84 = gdf.to_crs(epsg=4326)
                gdf['Latitude'] = gdf_wgs84.geometry.y.astype(float)
                gdf['Longitude'] = gdf_wgs84.geometry.x.astype(float)
            else:
                gdf['Latitude'] = gdf.geometry.y.astype(float)
                gdf['Longitude'] = gdf.geometry.x.astype(float)
                
            gdf['Altitude'] = 0.0

            temp_dir = tempfile.mkdtemp()
            output_base = os.path.join(temp_dir, "detected_palm_centers")
            gdf.to_file(output_base + '.shp', driver="ESRI Shapefile")

            zip_path = os.path.join(temp_dir, "detected_palm_centers.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    file_part = output_base + ext
                    if os.path.exists(file_part):
                        zipf.write(file_part, os.path.basename(file_part))

            col_a, col_b = st.columns([1, 2])
            with col_a:
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="💾 Download ArcMap Shapefile (.zip)",
                        data=f,
                        file_name="detected_palm_centers.zip",
                        mime="application/zip"
                    )
            with col_b:
                st.write("Attributes Preview:")
                st.dataframe(gdf[['Latitude', 'Longitude', 'Altitude']].head(10))

        else:
            st.warning("No tree crowns detected. Try lowering the Confidence Limit slider in the sidebar.")

    except Exception as e:
        st.error(f"An error occurred during detection: {e}")

    # Cleanup temporary local files
    if os.path.exists(temp_tif_path):
        os.remove(temp_tif_path)
    if os.path.exists(temp_jpg_path):
        os.remove(temp_jpg_path)
