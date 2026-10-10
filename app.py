"""
Streamlit Web Application for Worksheet Cleaner.
Mobile-first UI to upload photos, regenerate clean worksheet images using
Gemini Image models (Nano Banana / Flash Image), and compile them into a PDF.
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

# Always instantiate fresh cleaner instance to prevent stale cache on Cloud reloads
cleaner = WorksheetCleaner()

def main():
    st.title("📝 Worksheet Cleaner")
    st.markdown("Regenerate clean assignment images without answers — upload photos, get a clean printable PDF.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    cleaning_engine = st.sidebar.radio(
        "Cleaning Engine",
        ["🌟 AI Image Regeneration (Nano Banana / Flash-Image)", "🎨 Local OpenCV Filter (Offline)"],
        help="Gemini Image models take your photo and regenerate the exact clean worksheet image with answers erased."
    )

    api_key = ""
    chosen_model = ""
    if "AI Image Regeneration" in cleaning_engine:
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

        default_image_models = [
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image",
            "gemini-nano-banana-2.1",
            "gemini-3-pro-image",
        ]
        if hasattr(cleaner, "get_available_image_models"):
            try:
                image_models = cleaner.get_available_image_models(api_key)
            except Exception:
                image_models = default_image_models
        else:
            image_models = default_image_models

        chosen_model = st.sidebar.selectbox(
            "Image Generation Model",
            options=image_models,
            index=0,
            help="gemini-3.1-flash-lite-image is ultra-fast with high free-tier limits. Nano Banana Pro offers maximum precision."
        )

        with st.sidebar.expander("ℹ️ Model Details & Quotas"):
            st.markdown(
                """
                **⚡ `gemini-3.1-flash-lite-image`:**
                - High-speed, high-quota image generation model.
                - Best for multi-page batch booklet cleaning.

                **🍌 `gemini-3.1-flash-image` (Nano Banana 2):**
                - Standard balanced multimodal image model.

                **🧠 `gemini-3-pro-image` (Nano Banana Pro):**
                - Maximum fidelity for complex typography and tables.
                """
            )

        with st.sidebar.expander("✏️ View / Customize AI Prompt"):
            ai_prompt = st.text_area(
                "Prompt Instructions",
                value="Remove all handwritten pen answers from this image and return the clean questions only.",
                height=100,
                help="You can customize instructions here if you want specific rules for your worksheets."
            )
        default_ai_prompt = ai_prompt
    else:
        default_ai_prompt = ""
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

        if st.button("🧹 Regenerate Clean Pages & PDF", type="primary", use_container_width=True):
            if "AI Image Regeneration" in cleaning_engine and not api_key.strip():
                st.error("⚠️ Please enter your Gemini API Key in the sidebar to use AI mode. It's completely free from Google AI Studio!")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_images = []

            for i, file in enumerate(uploaded_files):
                file_bytes = file.getvalue()

                if "AI Image Regeneration" in cleaning_engine:
                    status_text.text(f"🎨 Regenerating clean image {i+1} of {len(uploaded_files)} using {chosen_model}...")
                    if hasattr(cleaner, "regenerate_image_with_ai"):
                        cleaned_bgr, msg = cleaner.regenerate_image_with_ai(file_bytes, api_key, chosen_model=chosen_model, custom_prompt=default_ai_prompt)
                    else:
                        from cleaner import WorksheetCleaner
                        cleaned_bgr, msg = WorksheetCleaner().regenerate_image_with_ai(file_bytes, api_key, chosen_model=chosen_model, custom_prompt=default_ai_prompt)
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
                    st.markdown(f"**Page {i+1}: {file.name}**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(file_bytes, caption="Original Uploaded Photo", use_container_width=True)
                    with col2:
                        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
                        st.image(cleaned_rgb, caption="✨ AI-Regenerated Clean Image", use_container_width=True)

                progress_bar.progress((i + 1) / len(uploaded_files))

            if processed_images:
                status_text.text("📄 Assembling clean images into printable PDF...")
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
            💡 <strong>Image Regeneration:</strong> Takes your solved booklet photo and regenerates a clean digital image with all answers removed, ready to print!
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
