import os
import tempfile
import zipfile
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from roboflow import Roboflow
from scipy.spatial import cKDTree
from shapely.geometry import Point

st.set_page_config(page_title="Oil Palm Center Mapping", layout="wide")

# Fixed model/inference defaults
API_KEY = "yVaMpDjeXPH2Mzqs41u7"
OVERLAP_SETTING = 50


# Cache model connection
@st.cache_resource
def load_roboflow_model(api_key, version_number):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("hanifs-workspace-bd93u").project(
        "oil-palm-tree-detection-sv9gl"
    )
    return project.version(version_number).model


# -----------------------------------------------------------------------------
# SIDEBAR - ONLY CONFIDENCE LIMIT
# -----------------------------------------------------------------------------
st.sidebar.header("AI Settings")
confidence_setting = st.sidebar.slider(
    "Confidence Limit (%)", min_value=1, max_value=100, value=5
)

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.title("🌴 Oil Palm Detection")
st.write(
    "Extract spatial metadata from GeoTIFF files and map palm crowns into a ready-to-export GIS shapefile package."
)

input_method = st.radio(
    "File Source:",
    [
        "Upload File (Small/Medium)",
        "Local File Path (Best for Large Files)",
    ],
)

tif_path = None
if input_method == "Upload File (Small/Medium)":
    uploaded_file = st.file_uploader(
        "Upload GeoTIFF (.tif)", type=["tif", "tiff"]
    )
    if uploaded_file is not None:
        temp_uploaded_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".tif"
        )
        while chunk := uploaded_file.read(8 * 1024 * 1024):
            temp_uploaded_file.write(chunk)
        temp_uploaded_file.close()
        tif_path = temp_uploaded_file.name
else:
    tif_path = st.text_input("Enter absolute file path (e.g. /content/501.tif)")

if tif_path and os.path.exists(tif_path):
    if st.button("🚀 Run AI Precision Pipeline"):
        with st.spinner(
            "Extracting spatial metadata and running inference..."
        ):
            try:
                model = load_roboflow_model(API_KEY, 10)

                # Open drone TIFF to capture real-world map coordinates
                with rasterio.open(tif_path) as src:
                    transform = src.transform
                    crs = src.crs
                    img_data = src.read()
                    if len(img_data.shape) == 3:
                        img_data = np.moveaxis(img_data, 0, -1)

                # Rescale image depth for AI
                if img_data.dtype != np.uint8:
                    img_data = cv2.normalize(
                        img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                    )
                if len(img_data.shape) == 2 or img_data.shape[-1] == 1:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
                elif img_data.shape[-1] > 3:
                    img_data = img_data[:, :, :3]

                # Save temporary image for prediction
                temp_dir = tempfile.mkdtemp()
                temp_jpg = os.path.join(temp_dir, "temp_ready.jpg")
                cv2.imwrite(
                    temp_jpg, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                )

                # Run inference using user's Confidence slider
                predictions = model.predict(
                    temp_jpg,
                    confidence=confidence_setting,
                    overlap=OVERLAP_SETTING,
                ).json()

                map_points = []
                pixel_coords = []

                if "predictions" in predictions:
                    for pred in predictions["predictions"]:
                        pixel_x = pred["x"]
                        pixel_y = pred["y"]

                        # Translate pixel spots directly to GIS map coordinates
                        map_x, map_y = transform * (pixel_x, pixel_y)
                        map_points.append(Point(map_x, map_y))
                        pixel_coords.append((pixel_x, pixel_y))

                st.session_state["map_points"] = map_points
                st.session_state["pixel_coords"] = pixel_coords
                st.session_state["crs"] = crs
                st.session_state["img_data"] = img_data
                st.session_state["processed_path"] = tif_path

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")

    # Display results if available for the current active file
    if (
        "map_points" in st.session_state
        and st.session_state.get("processed_path") == tif_path
    ):
        map_points = st.session_state["map_points"]
        pixel_coords = st.session_state["pixel_coords"]
        crs = st.session_state["crs"]
        img_data = st.session_state["img_data"]

        if len(map_points) > 0:
            st.success(f"🎉 Mapped {len(map_points)} Tree Crown Centers!")

            # -----------------------------------------------------------------
            # VISUAL PREVIEW
            # -----------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img_data)

            if pixel_coords:
                px_x, px_y = zip(*pixel_coords)
                ax.scatter(
                    px_x,
                    px_y,
                    c="#00FF00",
                    s=20,
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                )

            ax.set_title(
                f"AI Center Mapping Preview - Mapped {len(map_points)} Tree Crowns",
                fontsize=14,
                fontweight="bold",
            )
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

            # -----------------------------------------------------------------
            # SHAPEFILE & ATTRIBUTE TABLE EXPORT
            # -----------------------------------------------------------------
            gdf = gpd.GeoDataFrame(geometry=map_points, crs=crs)

            if crs is not None:
                gdf_wgs84 = gdf.to_crs(epsg=4326)
                gdf["Latitude"] = gdf_wgs84.geometry.y.astype(float)
                gdf["Longitude"] = gdf_wgs84.geometry.x.astype(float)
            else:
                gdf["Latitude"] = gdf.geometry.y.astype(float)
                gdf["Longitude"] = gdf.geometry.x.astype(float)

            gdf["Altitude"] = 0.0

            st.markdown("---")
            st.subheader("📋 Attribute Table Preview")
            st.dataframe(gdf[["Latitude", "Longitude", "Altitude"]].head(10))

            # Build export shapefile zip in a temporary directory
            zip_temp_dir = tempfile.mkdtemp()
            output_base = os.path.join(zip_temp_dir, "detected_palm_centers")
            gdf.to_file(output_base + ".shp", driver="ESRI Shapefile")

            zip_buffer_path = os.path.join(
                zip_temp_dir, "detected_palm_centers.zip"
            )
            with zipfile.ZipFile(zip_buffer_path, "w") as zipf:
                for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                    file_part = output_base + ext
                    if os.path.exists(file_part):
                        zipf.write(file_part, os.path.basename(file_part))

            with open(zip_buffer_path, "rb") as f:
                st.download_button(
                    label="💾 Download Shapefile Package (.zip)",
                    data=f.read(),
                    file_name="detected_palm_centers.zip",
                    mime="application/zip",
                )
        else:
            st.warning(
                "No palm crowns were detected. Try lowering the Confidence Limit in the sidebar."
            )
