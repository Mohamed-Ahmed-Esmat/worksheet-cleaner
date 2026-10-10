"""
Streamlit Web Application for Worksheet Cleaner.
Mobile-first UI to upload photos, remove pen ink, and generate a clean PDF.
"""

import streamlit as st
from PIL import Image
import io
import cv2
from cleaner import WorksheetCleaner

# Setup page configuration for mobile-friendly view
st.set_page_config(
    page_title="Worksheet Cleaner", 
    page_icon="📝", 
    layout="centered"
)

# Initialize cleaner
@st.cache_resource
def get_cleaner():
    return WorksheetCleaner()

cleaner = get_cleaner()

def main():
    st.title("📝 Worksheet Cleaner")
    st.markdown("Remove pen answers from worksheets — upload photos, get a clean PDF.")

    # Sidebar Settings
    st.sidebar.header("Settings")
    mode = st.sidebar.radio(
        "Detection Mode", 
        ["Blue Pen Removal", "Auto-Detect (Any Color)"],
        help="Blue Pen is faster and more accurate if answers are in blue."
    )
    
    sensitivity = st.sidebar.slider(
        "Sensitivity", 
        min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        help="Increase if some pen marks are left behind. Decrease if printed text gets erased."
    )
    
    enhance = st.sidebar.toggle("Enhance image quality", value=True)
    show_preview = st.sidebar.toggle("Show before/after preview", value=True)

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Worksheet Photos", 
        type=['jpg', 'jpeg', 'png', 'webp'], 
        accept_multiple_files=True,
        help="Select one or more photos of completed worksheets."
    )
    
    # Handle HEIC explicitly in text (Streamlit's uploader doesn't natively support HEIC without extra libs)
    st.caption("Note: HEIC images from iPhones should be converted to JPG/PNG before uploading.")

    if uploaded_files:
        if st.button("🧹 Clean & Generate PDF", type="primary", use_container_width=True):
            processed_images = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing image {i+1} of {len(uploaded_files)}...")
                
                # Read file bytes
                file_bytes = file.getvalue()
                
                # Process
                cleaned_bgr, msg = cleaner.process_image(
                    file_bytes, mode=mode, sensitivity=sensitivity, enhance=enhance
                )
                
                if cleaned_bgr is None:
                    st.error(f"File {file.name}: {msg}")
                    continue
                    
                if "Warning" in msg:
                    st.warning(f"File {file.name}: {msg}")
                    
                processed_images.append(cleaned_bgr)
                
                # Preview
                if show_preview:
                    st.markdown(f"**Preview: {file.name}**")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(file_bytes, caption="Original", use_container_width=True)
                        
                    with col2:
                        # Convert BGR to RGB for Streamlit display
                        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
                        st.image(cleaned_rgb, caption="Cleaned", use_container_width=True)
                        
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.text("Processing complete! Generating PDF...")
            
            if processed_images:
                pdf_bytes = cleaner.images_to_pdf(processed_images)
                
                st.success("✅ PDF ready for download!")
                st.download_button(
                    label="📄 Download Cleaned PDF",
                    data=pdf_bytes,
                    file_name="cleaned_worksheets.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("No images were successfully processed.")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
            <strong>Tips:</strong><br>
            Ensure good lighting when taking photos.<br>
            If printed text gets removed, lower the sensitivity.<br>
            If pen marks remain, try 'Auto-Detect' mode or increase sensitivity.
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
