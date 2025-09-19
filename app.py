import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd

st.title("💎 ASET Image Analyzer")

uploaded_file = st.file_uploader("Upload an ASET Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ---- Step 1: Read image ----
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---- Step 2: Background removal ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    diamond_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ---- Step 3: Crop diamond ----
    coords = cv2.findNonZero(diamond_mask)
    x, y, w, h = cv2.boundingRect(coords)
    img_cropped = img_rgb[y:y + h, x:x + w]
    diamond_mask_cropped = diamond_mask[y:y + h, x:x + w]

    # ---- Step 4: Convert to HSV ----
    hsv = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)

    # ---- Step 5: Define color ranges ----
    red1 = cv2.inRange(hsv, (0, 90, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 90, 50), (179, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    green_mask = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
    blue_mask = cv2.inRange(hsv, (90, 60, 40), (140, 255, 255))

    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
    gray_mask  = cv2.inRange(hsv, (0, 0, 80), (180, 50, 200))
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))

    # ---- Step 6: Count pixels ----
    diamond_area = np.count_nonzero(diamond_mask_cropped)
    red_count   = np.count_nonzero(cv2.bitwise_and(red_mask, diamond_mask_cropped))
    green_count = np.count_nonzero(cv2.bitwise_and(green_mask, diamond_mask_cropped))
    blue_count  = np.count_nonzero(cv2.bitwise_and(blue_mask, diamond_mask_cropped))
    black_count = np.count_nonzero(cv2.bitwise_and(black_mask, diamond_mask_cropped))
    gray_count  = np.count_nonzero(cv2.bitwise_and(gray_mask, diamond_mask_cropped))
    white_count = np.count_nonzero(cv2.bitwise_and(white_mask, diamond_mask_cropped))
    grey_count = black_count + gray_count + white_count

    percentages = {
        "Red":   100 * red_count / diamond_area,
        "Green": 100 * green_count / diamond_area,
        "Blue":  100 * blue_count / diamond_area,
        "Others": 100 * grey_count / diamond_area
    }

    # ---- Step 7: Overlay detected colors ----
    overlay = np.zeros_like(img_cropped)
    overlay[red_mask > 0]   = [255, 0, 0]
    overlay[green_mask > 0] = [0, 255, 0]
    overlay[blue_mask > 0]  = [0, 0, 255]
    overlay[black_mask > 0] = [128, 128, 128]
    overlay[gray_mask > 0]  = [128, 128, 128]
    overlay[white_mask > 0] = [128, 128, 128]
    blended = cv2.addWeighted(img_cropped, 0.6, overlay, 0.4, 0)

    # ---- Step 8: Visualization ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(img_rgb); axes[0, 0].set_title("Original ASET image"); axes[0, 0].axis('off')
    axes[0, 1].imshow(img_cropped); axes[0, 1].set_title("Diamond (background removed)"); axes[0, 1].axis('off')
    axes[1, 0].imshow(blended); axes[1, 0].set_title("Detected colors"); axes[1, 0].axis('off')
    colors = ['#FF0000', '#00FF00', '#0000FF', '#808080']
    axes[1, 1].pie(percentages.values(), labels=percentages.keys(), autopct='%1.1f%%', colors=colors)
    axes[1, 1].set_title("Color distribution inside diamond")
    plt.tight_layout()

    st.pyplot(fig)

    # ---- Step 9: Show percentages ----
    st.subheader("📊 Color Percentages:")
    for color, pct in percentages.items():
        st.write(f"**{color}:** {pct:.2f}%")

    # ---- Step 10: Download buttons ----
    # Save PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="📥 Download Analysis as PNG",
        data=buf.getvalue(),
        file_name="aset_analysis.png",
        mime="image/png"
    )

    # Save CSV
    df = pd.DataFrame(list(percentages.items()), columns=["Color", "Percentage"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="aset_results.csv",
        mime="text/csv"
    )

