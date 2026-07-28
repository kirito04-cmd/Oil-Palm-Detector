import streamlit as st
import os
import cv2
import numpy as np
import rasterio
import geopandas as gpd
import zipfile
import tempfile
import matplotlib.pyplot as plt
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

st.title("🌴 Oil Palm Tree Detection")
st.write("Upload your drone orthophoto (.tif), let the AI find the tree crowns, and download your sorted ArcMap-ready Shapefile.")

# Sidebar Settings
st.sidebar.header("AI Settings")
api_key = st.sidebar.text_input("Roboflow API Key", value="yVaMpDjeXPH2Mzqs41u7", type="password")
confidence_setting = st.sidebar.slider("Confidence Limit (%)", min_value=1, max_value=100, value=5)
overlap_setting = st.sidebar.slider("Overlap Limit (%)", min_value=1, max_value=100, value=50)

# File Uploader
uploaded_file = st.file_uploader("Upload Drone GeoTIFF Image (.tif)", type=["tif", "tiff"])

if uploaded_file is not None:
    # -----------------------------------------------------------------
    # STEP 1: INITIAL LOADING PROGRESS BAR
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

    load_progress.progress(75, text="🧠 Requesting YOLO model predictions from Roboflow...")

    try:
        model = load_roboflow_model(api_key=api_key, version_number=10)
        predictions = model.predict(temp_jpg_path, confidence=confidence_setting, overlap=overlap_setting).json()
        
        load_progress.progress(90, text="📍 Mapping coordinates and generating preview...")

        map_points = []
        pixel_coords = []

        if "predictions" in predictions:
            for pred in predictions["predictions"]:
                pixel_x, pixel_y = pred["x"], pred["y"]
                map_x, map_y = transform * (pixel_x, pixel_y)
                map_points.append(Point(map_x, map_y))
                pixel_coords.append((pixel_x, pixel_y))

        load_progress.progress(100, text="✅ Detection complete!")
        load_progress.empty()  # Clear loading bar once done

        # -----------------------------------------------------------------
        # STEP 2: DISPLAY RESULTS & EXPORT SHAPEFILE
        # -----------------------------------------------------------------
        if len(map_points) > 0:
            st.success(f"🎉 Mapped {len(map_points)} Tree Crowns successfully!")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("AI Detection Preview")
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(img_data)
                for px, py in pixel_coords:
                    ax.scatter(px, py, c='#00FF00', s=15, edgecolors='black', linewidths=1, zorder=5)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.subheader("Export GIS Shapefile")
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

                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="💾 Download ArcMap Shapefile (.zip)",
                        data=f,
                        file_name="detected_palm_centers.zip",
                        mime="application/zip"
                    )
                
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
