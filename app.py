# 1. Install dependencies
!pip install -q streamlit geopandas shapely rasterio pyngrok
!npm install -g localtunnel

# 2. Write the Streamlit application code
%%writefile app.py
import os
import shutil
import tempfile
import numpy as np
import geopandas as gpd
import streamlit as st
import rasterio
from shapely.geometry import Point

st.set_page_config(page_title="Orthophoto Detector", layout="wide")
st.title("🛰️ Large Orthophoto Feature Detector")

# Input for the Colab path
file_path = st.text_input(
    "Paste your Colab file path here:", 
    value="/content/your_orthophoto.tif",
    help="Right-click your file in the Colab sidebar and select 'Copy path'"
)

if st.button("Run Detection"):
    if not os.path.exists(file_path):
        st.error(f"File not found at `{file_path}`. Please check the path and try again.")
    else:
        with st.spinner("Reading geospatial metadata and detecting features..."):
            try:
                # Open GeoTIFF using rasterio
                with rasterio.open(file_path) as src:
                    bounds = src.bounds
                    crs = src.crs
                    
                    st.info(f"Loaded GeoTIFF | Resolution: {src.width}x{src.height} px | CRS: {crs}")
                    
                    # --- DETECTION LOGIC (Mock coordinates within image bounds) ---
                    np.random.seed(42)
                    random_xs = np.random.uniform(bounds.left, bounds.right, 5)
                    random_ys = np.random.uniform(bounds.bottom, bounds.top, 5)
                    confidences = np.random.uniform(0.85, 0.99, 5)
                    
                    # Create Spatial GeoDataFrame
                    geometry = [Point(x, y) for x, y in zip(random_xs, random_ys)]
                    gdf = gpd.GeoDataFrame(
                        {"id": range(1, 6), "confidence": np.round(confidences, 2)},
                        geometry=geometry,
                        crs=crs
                    )
                    
                    # Convert to standard lat/lon for output
                    if gdf.crs and gdf.crs != "EPSG:4326":
                        gdf = gdf.to_crs("EPSG:4326")

                st.success(f"Detected {len(gdf)} spatial points!")
                st.dataframe(gdf.drop(columns="geometry"))

                # Package Shapefile into ZIP
                with tempfile.TemporaryDirectory() as tmp_dir:
                    shp_path = os.path.join(tmp_dir, "detected_points.shp")
                    gdf.to_file(shp_path, driver="ESRI Shapefile")
                    
                    zip_archive = shutil.make_archive(
                        base_name=os.path.join(tmp_dir, "detected_points"),
                        format="zip",
                        root_dir=tmp_dir
                    )
                    
                    with open(zip_archive, "rb") as f:
                        zip_bytes = f.read()

                # Streamlit download button
                st.download_button(
                    label="📥 Download Shapefile (.zip)",
                    data=zip_bytes,
                    file_name="detected_points.zip",
                    mime="application/zip"
                )

            except Exception as e:
                st.error(f"Error processing image: {e}")
