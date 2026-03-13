import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import logic
import math
import numpy as np
import os
import requests

# 1. Page Config
st.set_page_config(page_title="SmartSpace", page_icon="⬜", layout="centered", initial_sidebar_state="expanded")

# 2. Styles
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e2e8f0; }
        .stApp { background-color: #0f172a; }
        h1 { color: #ffffff; }
        .stButton>button { background-color: #3b82f6; color: white; border-radius: 6px; padding: 12px; border: none; }
        .stFileUploader { background-color: #1e293b; padding: 2rem; border-radius: 12px; border: 1px dashed #475569;}
        header, footer {visibility: hidden;}
        .score-card {
            background: linear-gradient(135deg, #1e293b, #334155);
            border-radius: 12px; padding: 20px; margin: 10px 0;
            border: 1px solid #475569; text-align: center;
        }
        .score-number { font-size: 48px; font-weight: 700; }
        .score-good { color: #22c55e; }
        .score-mid { color: #f59e0b; }
        .score-bad { color: #ef4444; }
        .metric-row {
            display: flex; justify-content: space-around; flex-wrap: wrap;
            gap: 10px; margin: 10px 0;
        }
        .metric-box {
            background: #1e293b; border-radius: 8px; padding: 12px 16px;
            border: 1px solid #475569; text-align: center; flex: 1; min-width: 120px;
        }
        .metric-value { font-size: 24px; font-weight: 600; color: #60a5fa; }
        .metric-label { font-size: 12px; color: #94a3b8; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar - User Preferences
st.sidebar.markdown("## Room Preferences")
room_type = st.sidebar.selectbox("Room Type", ["Bedroom", "Living Room", "Study", "Studio Apartment", "Kids Room"])
style_pref = st.sidebar.selectbox("Style", ["Minimalist", "Cozy", "Modern", "Functional", "Traditional"])
priority = st.sidebar.selectbox("Priority", ["Better Sleep", "More Workspace", "Entertainment Area", "General Flow", "Storage"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Knowledge Base")
st.sidebar.info("Drop interior design PDFs into the `knowledge/` folder and restart the app to enhance AI suggestions.")

# 4. Global Download Utility
def download_model_if_missing(model_name, url):
    if not os.path.exists(model_name):
        st.warning(f"Downloading {model_name} (~{( '50MB' if 'yolo' in model_name else '668MB' )}). This may take a few minutes...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 # 1 Megabyte
            
            with open(model_name, 'wb') as file:
                downloaded_size = 0
                for data in response.iter_content(block_size):
                    file.write(data)
                    downloaded_size += len(data)
                    if total_size_in_bytes > 0:
                        progress = min(1.0, downloaded_size / total_size_in_bytes)
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded {downloaded_size//(1024*1024)}MB / {total_size_in_bytes//(1024*1024)}MB")
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"Downloaded {model_name} successfully!")
            
        except Exception as e:
            st.error(f"Failed to download {model_name}: {e}")
            if os.path.exists(model_name):
                os.remove(model_name)  # Clean up partial file

# Check and download models on startup
download_model_if_missing("yolov8m.pt", "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt")
download_model_if_missing("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# 5. Load YOLO Model
@st.cache_resource
def load_yolo():
    return YOLO("yolov8m.pt")

try:
    model = load_yolo()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 5. Grid Position Logic
def get_position_description(box, img_width, img_height):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    col = "Left" if cx < img_width / 3 else "Right" if cx > 2 * img_width / 3 else "Center"
    row = "Top" if cy < img_height / 3 else "Bottom" if cy > 2 * img_height / 3 else "Middle"
    if col == "Center" and row == "Middle":
        return "Center-Middle"
    return f"{row}-{col}"

def get_center(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return ((x1 + x2) / 2), ((y1 + y2) / 2)

def get_area(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (x2 - x1) * (y2 - y1)

# 6. IoU Collision Detection
def compute_iou(box1, box2):
    a = box1.xyxy[0].tolist()
    b = box2.xyxy[0].tolist()
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    if union == 0:
        return 0
    return intersection / union

# 7. Enhanced Spatial Analysis
def calculate_relationships(results, width, height):
    descriptions = []
    boxes = results[0].boxes
    names = model.names
    objects = []

    room_area = width * height

    # Zone density tracking (3x3 grid)
    zone_areas = {}
    zone_names = ["Top-Left", "Top-Center", "Top-Right",
                  "Middle-Left", "Center-Middle", "Middle-Right",
                  "Bottom-Left", "Bottom-Center", "Bottom-Right"]
    zone_cell_area = room_area / 9.0

    for zn in zone_names:
        zone_areas[zn] = 0.0

    total_furniture_area = 0

    # Pass 1: Grid Positions + Area Coverage
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        name = names[class_id].capitalize()
        position = get_position_description(box, width, height)
        area = get_area(box)
        area_pct = (area / room_area) * 100

        desc = f"The {name} is in the {position} zone, covering {area_pct:.1f}% of room area."
        descriptions.append(desc)

        cx, cy = get_center(box)
        objects.append({"name": name, "x": cx, "y": cy, "id": i, "box": box, "area": area})

        total_furniture_area += area
        if position in zone_areas:
            zone_areas[position] += area

    # Calculate zone densities as percentage
    zone_densities = {}
    for zn in zone_names:
        zone_densities[zn] = (zone_areas[zn] / zone_cell_area) * 100 if zone_cell_area > 0 else 0

    # Pass 2: Distances + Collisions
    collision_count = 0
    collision_details = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj1 = objects[i]
            obj2 = objects[j]

            dist = math.sqrt((obj1['x'] - obj2['x'])**2 + (obj1['y'] - obj2['y'])**2)
            normalized_dist = dist / width

            # IoU collision check
            iou = compute_iou(obj1['box'], obj2['box'])
            if iou > 0.05:
                rel = f"COLLISION: {obj1['name']} overlaps with {obj2['name']} (IoU={iou:.2f})."
                descriptions.append(rel)
                collision_count += 1
                collision_details.append(f"{obj1['name']} <-> {obj2['name']} (IoU={iou:.2f})")
            elif normalized_dist < 0.15:
                rel = f"The {obj1['name']} is touching or blocking the {obj2['name']}."
                descriptions.append(rel)
            elif normalized_dist < 0.30:
                rel = f"The {obj1['name']} is near the {obj2['name']}."
                descriptions.append(rel)

    # Build metrics dict
    metrics = {
        "total_coverage_pct": (total_furniture_area / room_area) * 100 if room_area > 0 else 0,
        "furniture_count": len(objects),
        "collision_count": collision_count,
        "collision_details": collision_details,
        "zone_densities": zone_densities,
        "objects": objects,
    }

    return descriptions, [obj['name'] for obj in objects], metrics

# 8. Heatmap Generation
def generate_heatmap(image, metrics, width, height):
    """Generate a color-coded zone density heatmap overlay."""
    heatmap = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(heatmap)

    zone_densities = metrics.get("zone_densities", {})
    cell_w = width // 3
    cell_h = height // 3

    zone_grid = [
        ["Top-Left", "Top-Center", "Top-Right"],
        ["Middle-Left", "Center-Middle", "Middle-Right"],
        ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
    ]

    for row_idx, row in enumerate(zone_grid):
        for col_idx, zone_name in enumerate(row):
            density = zone_densities.get(zone_name, 0)
            x1 = col_idx * cell_w
            y1 = row_idx * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            # Color: green (open) -> yellow (moderate) -> red (crowded)
            if density < 20:
                r, g, b = 34, 197, 94   # green
            elif density < 50:
                r, g, b = 245, 158, 11  # amber
            else:
                r, g, b = 239, 68, 68   # red

            alpha = min(120, int(density * 1.5) + 30)
            draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, alpha))

    # Composite heatmap onto original image
    base = image.convert('RGBA')
    combined = Image.alpha_composite(base, heatmap)
    return combined.convert('RGB')

# 9. Merge detections from multiple images
def merge_multi_image_results(all_results_data):
    """Merge detections from multiple images into one unified analysis."""
    merged_sentences = []
    merged_items = []
    merged_metrics = {
        "total_coverage_pct": 0,
        "furniture_count": 0,
        "collision_count": 0,
        "collision_details": [],
        "zone_densities": {},
        "objects": [],
    }

    zone_density_sums = {}
    num_images = len(all_results_data)

    for idx, data in enumerate(all_results_data):
        sentences, items, metrics = data
        img_label = f"[Image {idx + 1}]"

        for s in sentences:
            merged_sentences.append(f"{img_label} {s}")
        merged_items.extend(items)

        merged_metrics["collision_count"] += metrics["collision_count"]
        merged_metrics["collision_details"].extend(metrics["collision_details"])

        # Average zone densities across images
        for zone, density in metrics.get("zone_densities", {}).items():
            if zone not in zone_density_sums:
                zone_density_sums[zone] = 0
            zone_density_sums[zone] += density

    # Average the zone densities
    for zone in zone_density_sums:
        merged_metrics["zone_densities"][zone] = zone_density_sums[zone] / num_images

    # Deduplicate items but keep count
    item_counts = {}
    for item in merged_items:
        item_counts[item] = item_counts.get(item, 0) + 1

    # Use max coverage across images (most complete view)
    coverages = [d[2].get("total_coverage_pct", 0) for d in all_results_data]
    merged_metrics["total_coverage_pct"] = max(coverages) if coverages else 0
    merged_metrics["furniture_count"] = len(set(merged_items))

    # Add count info to sentences
    for item, count in item_counts.items():
        if count > 1:
            merged_sentences.append(f"{item} detected in {count} images (confirmed presence).")

    return merged_sentences, list(set(merged_items)), merged_metrics


# 10. UI Layout
st.markdown("# SmartSpace Studio")
st.markdown("### Interior Layout Analysis & Optimization System")

uploaded_files = st.file_uploader(
    "Upload Room Images (1-3 photos for best results)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 3:
        st.warning("Maximum 3 images allowed. Using the first 3.")
        uploaded_files = uploaded_files[:3]

    st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")

    # Display uploaded images side by side
    cols = st.columns(len(uploaded_files))
    images = []
    for idx, (col, uploaded_file) in enumerate(zip(cols, uploaded_files)):
        image = Image.open(uploaded_file)
        images.append(image)
        col.image(image, caption=f"Image {idx + 1}", use_column_width=True)

    if st.button("Initialize Spatial Analysis"):
        with st.spinner("Analyzing spatial geometry across {} image(s)...".format(len(images))):

            all_results_data = []
            all_yolo_plots = []

            for idx, image in enumerate(images):
                w, h = image.size
                results = model(image, classes=[56, 57, 59, 60], conf=0.25)
                res_plotted = results[0].plot(line_width=2, font_size=1)
                all_yolo_plots.append(res_plotted)

                sentences, items, metrics = calculate_relationships(results, w, h)
                all_results_data.append((sentences, items, metrics))

            # Show YOLO detection results
            st.markdown("### Detected Spatial Elements")
            det_cols = st.columns(len(all_yolo_plots))
            for idx, (col, plot) in enumerate(zip(det_cols, all_yolo_plots)):
                col.image(plot, caption=f"Detection {idx + 1}", use_column_width=True)

            # Merge or use single image results
            if len(all_results_data) == 1:
                spatial_sentences, item_list, final_metrics = all_results_data[0]
            else:
                spatial_sentences, item_list, final_metrics = merge_multi_image_results(all_results_data)

            # Generate and show heatmap (use first/largest image as base)
            st.markdown("### Room Utilization Heatmap")
            st.caption("Green = Open space | Amber = Moderate | Red = Overcrowded")
            main_image = images[0]
            heatmap_img = generate_heatmap(main_image, final_metrics, main_image.size[0], main_image.size[1])
            st.image(heatmap_img, caption="Zone Density Heatmap", use_column_width=True)

            # Show metrics dashboard
            coverage = final_metrics.get("total_coverage_pct", 0)
            furniture_count = final_metrics.get("furniture_count", 0)
            collision_count = final_metrics.get("collision_count", 0)

            st.markdown("### Spatial Metrics")
            m_cols = st.columns(3)
            m_cols[0].metric("Furniture Coverage", f"{coverage:.1f}%")
            m_cols[1].metric("Pieces Detected", str(furniture_count))
            m_cols[2].metric("Collisions", str(collision_count))

            with st.expander("See AI Spatial Reasoning"):
                for sentence in spatial_sentences:
                    st.write(f"- {sentence}")

                st.markdown("**Zone Densities:**")
                for zone, density in final_metrics.get("zone_densities", {}).items():
                    bar_len = int(density / 5)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    st.text(f"  {zone:16s} [{bar}] {density:.0f}%")

        if spatial_sentences:
            st.markdown("---")
            st.markdown("### AI Customized Recommendations")

            preferences = {
                "room_type": room_type,
                "style": style_pref,
                "priority": priority,
            }

            with st.spinner("Synthesizing expert advice with RAG..."):
                advice, space_score = logic.get_design_suggestions(
                    spatial_sentences, item_list,
                    metrics=final_metrics,
                    preferences=preferences
                )

                # Show Space Efficiency Score
                if space_score is not None:
                    score_class = "score-good" if space_score >= 70 else "score-mid" if space_score >= 40 else "score-bad"
                    score_label = "Excellent" if space_score >= 70 else "Needs Work" if space_score >= 40 else "Poor"
                    st.markdown(f"""
                        <div class="score-card">
                            <div class="metric-label">SPACE EFFICIENCY SCORE</div>
                            <div class="score-number {score_class}">{space_score}</div>
                            <div style="color: #94a3b8;">{score_label}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.success(advice)

                st.markdown("---")
                st.caption(f"Analysis based on: {room_type} | Style: {style_pref} | Priority: {priority}")
        else:
            st.warning("No furniture detected. Try a clearer angle or check lighting.")
