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

# 1. Page Configuration
st.set_page_config(
    page_title="PalmPrecise AI | GIS",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern sleek look
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    </style>
""", unsafe_allow_html=True)

# 2. Header / Title Area
st.title("🌴 PalmPrecise AI")
st.subheader("Enterprise Oil Palm Tree Detection & Geospatial Exporter")
st.markdown("---")

# 3. Sidebar Settings
st.sidebar.image("https://img.icons8.com/clouds/100/000000/palm-tree.png", width=80)
st.sidebar.header("🕹️ Control Center")

api_key = st.sidebar.text_input("Roboflow API Key", value="yVaMpDjeXPH2Mzqs41u7", type="password")
confidence_setting = st.sidebar.slider("AI Confidence Threshold (%)", min_value=1, max_value=100, value=25)
overlap_setting = st.sidebar.slider("IoU Overlap Limit (%)", min_value=1, max_value=100, value=30)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip:** Lower the confidence threshold if the AI is missing very small, newly born tree crowns.")

# 4. Main App Layout - File Uploader
uploaded_file = st.file_uploader("📂 Drag and drop your Drone GeoTIFF Image (.tif)", type=["tif", "tiff"])

if uploaded_file is not None:
    # Setup temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_tif:
        temp_tif.write(uploaded_file.read())
        temp_tif_path = temp_tif.name

    # Read spatial coordinates
    with rasterio.open(temp_tif_path) as src:
        transform = src.transform
        crs = src.crs
        img_data = src.read()
        if len(img_data.shape) == 3:
            img_data = np.moveaxis(img_data, 0, -1)

    # Image Normalization
    if img_data.dtype != np.uint8:
        img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if len(img_data.shape) == 2 or img_data.shape[-1] == 1:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
    elif img_data.shape[-1] > 3:
        img_data = img_data[:, :, :3]

    temp_jpg_path = os.path.join(tempfile.gettempdir(), "temp_ready.jpg")
    cv2.imwrite(temp_jpg_path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))

    try:
        # Run AI Inference
        rf = Roboflow(api_key=api_key)
        # Note: Change version(10) to your new version number once you retrain!
        project = rf.workspace("hanifs-workspace-bd93u").project("oil-palm-tree-detection-sv9gl")
        model = project.version(10).model

        with st.spinner("🧠 AI is analyzing canopy imagery for young & mature stands..."):
            predictions = model.predict(temp_jpg_path, confidence=confidence_setting, overlap=overlap_setting).json()

        map_points = []
        pixel_coords = []

        if "predictions" in predictions:
            for pred in predictions["predictions"]:
                pixel_x, pixel_y = pred["x"], pred["y"]
                map_x, map_y = transform * (pixel_x, pixel_y)
                map_points.append(Point(map_x, map_y))
                pixel_coords.append((pixel_x, pixel_y))

        # 📊 EXECUTIVE DASHBOARD TABS
        tab1, tab2, tab3 = st.tabs(["📊 Analytics Summary", "🗺️ Interactive Canvas", "💾 GIS Export"])

        with tab1:
            st.markdown("### 📈 Inventory Insights")
            # Scorecard Metrics
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(label="Total Trees Detected", value=f"{len(map_points)} Stand(s)", delta="AI Verified")
            kpi2.metric(label="Coordinate Reference System (CRS)", value=str(crs.to_epsg()) if crs else "Local", delta="Geo-Referenced")
            kpi3.metric(label="Processing Status", value="Success ✨", delta="Done")
            
            st.markdown("---")
            if len(map_points) > 0:
                gdf = gpd.GeoDataFrame(geometry=map_points, crs=crs)
                gdf_wgs84 = gdf.to_crs(epsg=4326)
                gdf['Latitude'] = gdf_wgs84.geometry.y.astype(float)
                gdf['Longitude'] = gdf_wgs84.geometry.x.astype(float)
                
                st.write("**Top 5 Stand Spatial Coordinates Preview:**")
                st.dataframe(gdf[['Latitude', 'Longitude']].head(5), use_container_width=True)

        with tab2:
            st.markdown("### 🔍 Computer Vision Map Preview")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img_data)
            for px, py in pixel_coords:
                ax.scatter(px, py, c='#00FF00', s=20, edgecolors='black', linewidths=0.75, zorder=5)
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)

        with tab3:
            st.markdown("### 📥 Download GIS Deliverables")
            if len(map_points) > 0:
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

                st.info("Your shapefile payload includes standard spatial indexers (`.shp`, `.shx`, `.dbf`, `.prj`) perfectly compiled for ArcMap/QGIS.")
                
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="💾 Download ArcMap Shapefile (.zip)",
                        data=f,
                        file_name="detected_palm_centers.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

    except Exception as e:
        st.error(f"Error during runtime execution: {e}")

    # Cleanup Files
    if os.path.exists(temp_tif_path): os.remove(temp_tif_path)
    if os.path.exists(temp_jpg_path): os.remove(temp_jpg_path)
