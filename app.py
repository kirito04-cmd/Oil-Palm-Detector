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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_tif:
        temp_tif.write(uploaded_file.read())
        temp_tif_path = temp_tif.name

    st.info("🔄 Reading spatial coordinates from image metadata...")

    with rasterio.open(temp_tif_path) as src:
        transform = src.transform
        crs = src.crs
        img_data = src.read()
        if len(img_data.shape) == 3:
            img_data = np.moveaxis(img_data, 0, -1)

    if img_data.dtype != np.uint8:
        img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if len(img_data.shape) == 2 or img_data.shape[-1] == 1:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
    elif img_data.shape[-1] > 3:
        img_data = img_data[:, :, :3]

    temp_jpg_path = os.path.join(tempfile.gettempdir(), "temp_ready.jpg")
    cv2.imwrite(temp_jpg_path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("hanifs-workspace-bd93u").project("oil-palm-tree-detection-sv9gl")
        model = project.version(10).model

        with st.spinner("🧠 Running YOLOv11 Model Inference... Please wait."):
            predictions = model.predict(temp_jpg_path, confidence=confidence_setting, overlap=overlap_setting).json()

        map_points = []
        pixel_coords = []

        if "predictions" in predictions:
            for pred in predictions["predictions"]:
                pixel_x, pixel_y = pred["x"], pred["y"]
                map_x, map_y = transform * (pixel_x, pixel_y)
                map_points.append(Point(map_x, map_y))
                pixel_coords.append((pixel_x, pixel_y))

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

        with col2:
            st.subheader("Export GIS Shapefile")
            if len(map_points) > 0:
                gdf = gpd.GeoDataFrame(geometry=map_points, crs=crs)
                
                gdf_wgs84 = gdf.to_crs(epsg=4326)
                gdf['Latitude'] = gdf_wgs84.geometry.y.astype(float)
                gdf['Longitude'] = gdf_wgs84.geometry.x.astype(float)
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

    except Exception as e:
        st.error(f"Error during execution: {e}")

    if os.path.exists(temp_tif_path):
        os.remove(temp_tif_path)
    if os.path.exists(temp_jpg_path):
        os.remove(temp_jpg_path)
