"""
Streamlit Web Application for Worksheet Cleaner.
Mobile-first UI to upload photos, remove pen answers using Gemini AI or OpenCV,
and generate a clean printable PDF.
"""

import streamlit as st
import os
import cv2
from cleaner import WorksheetCleaner

# Setup page configuration for mobile-friendly view
st.set_page_config(
    page_title="Worksheet Cleaner",
    page_icon="📝",
    layout="centered"
)

# Custom mobile-friendly styling
st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    .preview-box {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        background-color: #ffffff;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def get_cleaner():
    return WorksheetCleaner()

cleaner = get_cleaner()

def main():
    st.title("📝 Worksheet Cleaner")
    st.markdown("Remove student answers from school booklets — upload photos, get a clean printable PDF.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    cleaning_engine = st.sidebar.radio(
        "Engine",
        ["🌟 Gemini AI (Recommended)", "🎨 Local OpenCV (Offline)"],
        help="Gemini AI completely erases handwriting, watermarks, and shadows, reconstructing a clean blank sheet."
    )

    api_key = ""
    if "Gemini AI" in cleaning_engine:
        # Check secrets / environment first
        default_key = os.environ.get("GEMINI_API_KEY", "")
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            default_key = st.secrets["GEMINI_API_KEY"]

        api_key = st.sidebar.text_input(
            "Gemini API Key",
            value=default_key,
            type="password",
            help="Free key from Google AI Studio (no credit card required)."
        )
        st.sidebar.markdown(
            "👉 [Get a free key at Google AI Studio](https://aistudio.google.com/apikey)",
            unsafe_allow_html=True
        )

    else:
        cv_mode = st.sidebar.radio(
            "Detection Mode",
            ["Blue Pen Removal", "Auto-Detect (Any Color)"],
            help="Filters out colored ink based on pixel thresholds."
        )
        sensitivity = st.sidebar.slider(
            "Sensitivity",
            min_value=0.5, max_value=2.0, value=1.0, step=0.1
        )
        enhance = st.sidebar.toggle("Enhance image quality", value=True)

    show_preview = st.sidebar.toggle("Show before / after preview", value=True)

    # Main Uploader
    uploaded_files = st.file_uploader(
        "Upload Worksheet Photos",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True,
        help="Take photos of completed booklet pages and upload them all together."
    )

    if uploaded_files:
        st.caption(f"📁 {len(uploaded_files)} image(s) selected")

        if st.button("🧹 Clean & Generate PDF", type="primary", use_container_width=True):
            if "Gemini AI" in cleaning_engine and not api_key.strip():
                st.error("⚠️ Please enter your Gemini API Key in the sidebar to use AI mode. It's completely free from Google AI Studio!")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            # -------------------------------------------------
            # 1. GEMINI AI MODE
            # -------------------------------------------------
            if "Gemini AI" in cleaning_engine:
                pages_html = []

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"🤖 AI processing page {i+1} of {len(uploaded_files)}...")
                    file_bytes = file.getvalue()

                    html_page, msg = cleaner.clean_with_gemini(file_bytes, api_key)

                    if not html_page:
                        st.error(f"Page {file.name}: {msg}")
                        continue

                    pages_html.append(html_page)

                    if show_preview:
                        st.markdown(f"#### 📄 Page {i+1}: {file.name}")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.caption("Original Photo")
                            st.image(file_bytes, use_container_width=True)
                        with col2:
                            st.caption("Clean Reconstructed Page")
                            with st.container():
                                st.markdown(
                                    f"<div class='preview-box'>{html_page}</div>",
                                    unsafe_allow_html=True
                                )

                    progress_bar.progress((i + 1) / len(uploaded_files))

                if pages_html:
                    status_text.text("✨ Generating high-resolution PDF...")
                    pdf_bytes = cleaner.html_to_pdf(pages_html)

                    st.success("✅ Clean PDF generated successfully!")
                    st.download_button(
                        label="📄 Download Cleaned PDF",
                        data=pdf_bytes,
                        file_name="clean_worksheets.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error("No pages were successfully processed.")

            # -------------------------------------------------
            # 2. LOCAL OPENCV MODE
            # -------------------------------------------------
            else:
                processed_images = []

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {i+1} of {len(uploaded_files)}...")
                    file_bytes = file.getvalue()

                    cleaned_bgr, msg = cleaner.process_image(
                        file_bytes, mode=cv_mode, sensitivity=sensitivity, enhance=enhance
                    )

                    if cleaned_bgr is None:
                        st.error(f"File {file.name}: {msg}")
                        continue

                    if "Warning" in msg:
                        st.warning(f"File {file.name}: {msg}")

                    processed_images.append(cleaned_bgr)

                    if show_preview:
                        st.markdown(f"**Preview: {file.name}**")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.image(file_bytes, caption="Original", use_container_width=True)
                        with col2:
                            cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
                            st.image(cleaned_rgb, caption="Cleaned", use_container_width=True)

                    progress_bar.progress((i + 1) / len(uploaded_files))

                if processed_images:
                    status_text.text("Processing complete! Generating PDF...")
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
        <div style='text-align: center; color: gray; font-size: 0.85em;'>
            💡 <strong>Pro Tip:</strong> Gemini AI mode completely removes watermark overlays, desk shadows, and handwriting while leaving pristine blank exercise sheets for your little bro to re-solve!
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
