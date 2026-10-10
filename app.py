"""
Streamlit Web Application for Worksheet Cleaner.
Mobile-first UI to upload photos, remove pen answers directly from the images,
and generate a clean printable PDF of the cleaned page images.
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
    st.markdown("Erase pen answers from school booklet photos — get back the exact clean page images in a PDF.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    cleaning_engine = st.sidebar.radio(
        "Cleaning Engine",
        ["🌟 Gemini AI Eraser (Recommended)", "🎨 Local OpenCV Filter (Offline)"],
        help="Gemini AI detects exact handwriting coordinates and erases only the answers while keeping the original page image intact."
    )

    api_key = ""
    chosen_model = ""
    if "Gemini AI" in cleaning_engine:
        default_key = os.environ.get("GEMINI_API_KEY", "")
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            default_key = st.secrets["GEMINI_API_KEY"]

        api_key = st.sidebar.text_input(
            "Gemini API Key",
            value=default_key,
            type="password",
            help="Free API key from Google AI Studio."
        )
        st.sidebar.markdown(
            "👉 [Get a free key at Google AI Studio](https://aistudio.google.com/apikey)",
            unsafe_allow_html=True
        )

        default_models = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-3.8-flash", "gemini-2.5-pro", "gemini-3.0-pro"]
        available_models = default_models

        if api_key.strip():
            fetched = cleaner.get_available_models(api_key)
            if fetched:
                available_models = fetched

        chosen_model = st.sidebar.selectbox(
            "AI Model",
            options=available_models,
            index=0,
            help="Flash-Lite is ultra-fast, lightweight, and generous with limits."
        )

        with st.sidebar.expander("📊 Model Tiers & Free Limits"):
            st.markdown(
                """
                **⚡ Gemini 3.1 Flash-Lite (Recommended):**
                - **Speed:** Fastest latency (~1s per image)
                - **Limit:** ~15–30 Requests/Min, ~1,500+ Requests/Day
                - **Cost:** 100% Free
                
                **🚀 Gemini 2.5 / 3.8 Flash:**
                - Standard high-capacity multimodal model.
                
                **🧠 Gemini 2.5 / 3.0 Pro:**
                - Deep reasoning for very messy or faint handwriting (~2–5 RPM).
                """
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
            processed_images = []

            for i, file in enumerate(uploaded_files):
                file_bytes = file.getvalue()

                # Process depending on engine
                if "Gemini AI" in cleaning_engine:
                    status_text.text(f"🤖 AI erasing answers from image {i+1} of {len(uploaded_files)} using {chosen_model}...")
                    cleaned_bgr, msg = cleaner.clean_image_with_gemini_ai(file_bytes, api_key, chosen_model=chosen_model)
                else:
                    status_text.text(f"Processing image {i+1} of {len(uploaded_files)}...")
                    cleaned_bgr, msg = cleaner.process_image(file_bytes, mode=cv_mode, sensitivity=sensitivity, enhance=enhance)

                if cleaned_bgr is None:
                    st.error(f"File {file.name}: {msg}")
                    continue

                if "Warning" in msg:
                    st.warning(f"File {file.name}: {msg}")

                processed_images.append(cleaned_bgr)

                # Show Side-by-Side Image Preview
                if show_preview:
                    st.markdown(f"**Preview: {file.name}**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(file_bytes, caption="Original Photo", use_container_width=True)
                    with col2:
                        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
                        st.image(cleaned_rgb, caption="Cleaned Image (Answers Erased)", use_container_width=True)

                progress_bar.progress((i + 1) / len(uploaded_files))

            if processed_images:
                status_text.text("✨ Compiling cleaned page images into PDF...")
                pdf_bytes = cleaner.images_to_pdf(processed_images)

                st.success("✅ Clean PDF ready for download!")
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
            💡 <strong>How it works:</strong> The AI precisely locates handwritten pen marks and erases them directly from your photos, preserving the original booklet layout, typography, and page numbers.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
