import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configure the Streamlit page
st.set_page_config(page_title="Image Processing", layout="wide")
st.title(" Image Processing Project")

# Custom CSS for sidebar styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(45deg, #1a1a1a, #2a2a2a) !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Upload Image
with st.sidebar.expander(" Upload Image", expanded=True):
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    # Convert uploaded file to an OpenCV-compatible format
    image = np.array(Image.open(uploaded_file))

    # Create two-column layout for display
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("###  Original Image")
        st.image(image, use_container_width=True)


    # Sidebar - Processing options
    with st.sidebar:
        st.markdown("## 🎚 Processing Controls")
        processor = st.radio("Select Operation:", [
            "Smoothing Filters", "Sharpening Filters", 
            "Edge Detection", "Rotate", "Grayscale"
        ])

    # Image processing logic
    processed = image.copy()  # Create a copy for modification

    with col2:
        if processor == "Smoothing Filters":
            smooth_type = st.radio("Filter Type:", ["Gaussian", "Median", "Bilateral"])
            kernel_size = st.slider("Kernel Size", 3, 25, 9, 2)

            if smooth_type == "Gaussian":
                sigma = st.slider("Sigma", 0.1, 5.0, 1.5)
                processed = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
            elif smooth_type == "Median":
                processed = cv2.medianBlur(image, kernel_size)
            
            else:
                d = st.slider("Diameter", 1, 15, 9)
                sigma_color = st.slider("Color Sigma", 1, 200, 75)
                sigma_space = st.slider("Spatial Sigma", 1, 200, 75)
                processed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        elif processor == "Sharpening Filters":
            sharp_type = st.radio("Technique:", ["Laplacian", "Unsharp Mask"])

            if sharp_type == "Laplacian":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                processed = cv2.filter2D(image, -1, kernel)
            
            else:
                blur = cv2.GaussianBlur(image, (0,0), 3)
                processed = cv2.addWeighted(image, 1.5, blur, -0.5, 0)

        elif processor == "Edge Detection":
            edge_type = st.radio("Detection Method:", ["Canny", "Sobel", "Laplacian"])
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if edge_type == "Canny":
                threshold1 = st.slider("Low Threshold", 0, 255, 50)
                threshold2 = st.slider("High Threshold", 0, 255, 150)
                processed = cv2.Canny(gray, threshold1, threshold2)

            elif edge_type == "Sobel":
                dx = st.slider("X Derivative", 0, 2, 1)
                dy = st.slider("Y Derivative", 0, 2, 1)
                sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=5)
                processed = np.uint8(np.absolute(sobel))

            else:
                processed = cv2.Laplacian(gray, cv2.CV_64F)
                processed = np.uint8(np.absolute(processed))

        elif processor == "Rotate":
            angle = st.slider("Rotation Angle", -180, 180, 0)
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            processed = cv2.warpAffine(image, rot_matrix, (w, h))

        elif processor == "Grayscale":
            processed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Display the processed image
        st.markdown("###  Processed Image")
        st.image(processed, use_container_width=True, channels="RGB" if len(processed.shape) == 3 else "GRAY")


        # Convert processed image to downloadable format
        processed_pil = Image.fromarray(processed)
        st.download_button(
            label=" Download Processed Image",
            data=cv2.imencode(".png", np.array(processed_pil))[1].tobytes(),
            file_name="processed_image.png",
            mime="image/png"
        )

else:
    # Placeholder for when no image is uploaded
    st.markdown("""
        <div style="text-align: center; padding: 100px 20px">
            <h2 style="color: #666">📁 Drag & Drop Image to Begin</h2>
            <p style="color: #444">Supports JPG, PNG, JPEG formats</p>
        </div>
    """, unsafe_allow_html=True)
