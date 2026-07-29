import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
from shapely.geometry import Point
from PIL import Image

st.set_page_config(page_title="Detection & Shapefile Exporter", layout="wide")

st.title("🛰️ Object Detection & Shapefile Exporter")
st.write("Upload an image, detect features, and download the resulting point spatial dataset as a Shapefile.")

# 1. File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Run Detection"):
        with st.spinner("Processing image and extracting point coordinates..."):
            
            # --- REPLACE THIS SECTION WITH YOUR ACTUAL DETECTION MODEL ---
            # Simulating detection output: generating 5 mock (longitude, latitude) points
            # or (X, Y) pixel coordinates
            np.random.seed(42)
            img_width, img_height = image.size
            
            # Example coordinates (adjust CRS to EPSG:4326 for lat/lon or EPSG:3857)
            random_lons = np.random.uniform(-122.45, -122.40, 5)
            random_lats = np.random.uniform(37.75, 37.80, 5)
            confidences = np.random.uniform(0.75, 0.98, 5)
            labels = ["detected_object"] * 5
            
            # -------------------------------------------------------------
            
            # Create GeoDataFrame
            geometry = [Point(lon, lat) for lon, lat in zip(random_lons, random_lats)]
            gdf = gpd.GeoDataFrame(
                {
                    "id": range(1, len(geometry) + 1),
                    "label": labels,
                    "confidence": np.round(confidences, 2)
                },
                geometry=geometry,
                crs="EPSG:4326"  # Standard WGS84 projection
            )
            
            st.success(f"Detected {len(gdf)} points!")
            st.dataframe(gdf.drop(columns="geometry"))

            # Save Shapefile into a temporary directory to keep it clean
            with tempfile.TemporaryDirectory() as tmp_dir:
                shp_base_name = "detected_points"
                shp_path = os.path.join(tmp_dir, f"{shp_base_name}.shp")
                zip_path_without_ext = os.path.join(tmp_dir, "detected_points")
                
                # Save Shapefile components (.shp, .shx, .dbf, .prj)
                gdf.to_file(shp_path, driver="ESRI Shapefile")
                
                # Compress all shapefile components into a single ZIP archive
                zip_archive = shutil.make_archive(
                    base_name=zip_path_without_ext,
                    format="zip",
                    root_dir=tmp_dir
                )
                
                # Read the zipped file into memory for downloading
                with open(zip_archive, "rb") as f:
                    zip_bytes = f.read()
            
            # Streamlit Download Button
            st.download_button(
                label="📥 Download Points Shapefile (.zip)",
                data=zip_bytes,
                file_name="detected_points.zip",
                mime="application/zip"
            )
