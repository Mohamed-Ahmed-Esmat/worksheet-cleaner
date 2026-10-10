"""
Streamlit Web Application for Worksheet Cleaner.
Mobile-first UI to upload photos, regenerate clean worksheet images using
OpenRouter (Nano Banana Pro / Gemini Image), and compile them into a PDF.
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

# Fresh instance to prevent stale cache across reloads
cleaner = WorksheetCleaner()

def main():
    st.title("📝 Worksheet Cleaner")
    st.markdown("Regenerate clean assignment images without answers — upload photos, get a clean printable PDF.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    cleaning_engine = st.sidebar.radio(
        "Cleaning Engine",
        ["🌟 OpenRouter (Nano Banana Pro)", "🎨 Local OpenCV Filter (Offline)"],
        help="OpenRouter routes directly to Google's Nano Banana Pro with zero regional geo-blocks."
    )

    api_key = ""
    chosen_model = ""
    default_ai_prompt = ""

    if "OpenRouter" in cleaning_engine:
        # Check secrets / environment first
        default_key = os.environ.get("OPENROUTER_API_KEY", "")
        if hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
            default_key = st.secrets["OPENROUTER_API_KEY"]

        api_key = st.sidebar.text_input(
            "OpenRouter API Key",
            value=default_key,
            type="password",
            help="Your API key from openrouter.ai (starts with sk-or-v1-...)."
        )
        st.sidebar.markdown(
            "👉 [Get API Key at openrouter.ai/keys](https://openrouter.ai/keys)",
            unsafe_allow_html=True
        )

        model_options = {
            "🍌 Nano Banana Pro (google/gemini-3-pro-image) [Recommended]": "google/gemini-3-pro-image",
            "⚡ Nano Banana 2.1 (google/gemini-nano-banana-2.1) [Fast & Cheap]": "google/gemini-nano-banana-2.1",
            "⚡ Gemini 3.1 Flash Image (google/gemini-3.1-flash-image)": "google/gemini-3.1-flash-image",
            "⚡ Gemini 2.5 Flash Image (google/gemini-2.5-flash-image)": "google/gemini-2.5-flash-image",
            "✨ OpenAI GPT-5 Image (openai/gpt-5-image)": "openai/gpt-5-image",
            "✨ OpenAI GPT-5 Image Mini (openai/gpt-5-image-mini)": "openai/gpt-5-image-mini",
            "🎨 FLUX 2 Pro (black-forest-labs/flux.2-pro) [Text-to-Image Only]": "black-forest-labs/flux.2-pro",
            "🎨 FLUX 1 Dev (black-forest-labs/flux-1-dev) [Text-to-Image Only]": "black-forest-labs/flux-1-dev",
            "🎨 FLUX 1 Schnell (black-forest-labs/flux-1-schnell) [Text-to-Image Only]": "black-forest-labs/flux-1-schnell",
            "👁️ Qwen 2.5 VL 72B (qwen/qwen-2.5-vl-72b-instruct) [Vision-to-Text Only]": "qwen/qwen-2.5-vl-72b-instruct",
            "👁️ Qwen 3 VL 8B (qwen/qwen3-vl-8b-instruct) [Vision-to-Text Only]": "qwen/qwen3-vl-8b-instruct",
            "✏️ Custom Model ID...": "custom",
        }

        selected_label = st.sidebar.selectbox(
            "AI Model",
            options=list(model_options.keys()),
            index=0,
            help="Select the AI model. For worksheets, use Image-to-Image models (Nano Banana Pro / Gemini / GPT-5 Image)."
        )

        if model_options[selected_label] == "custom":
            chosen_model = st.sidebar.text_input("Enter OpenRouter Model Slug", value="google/gemini-3-pro-image")
        else:
            chosen_model = model_options[selected_label]

        with st.sidebar.expander("ℹ️ Model Info & Modalities"):
            st.markdown(
                """
                **🍌 Google Image Models (Native Image-to-Image):**
                - `google/gemini-3-pro-image` (Nano Banana Pro — Highest quality text & question preservation)
                - `google/gemini-nano-banana-2.1` (Ultra fast & cheap)
                - `google/gemini-3.1-flash-image` (Flash image generation)
                - `google/gemini-2.5-flash-image` (Standard Flash image)

                **✨ OpenAI Image Models:**
                - `openai/gpt-5-image` (Flagship multimodal image output)
                - `openai/gpt-5-image-mini` (Fast multimodal image output)

                **🎨 FLUX Models (Black Forest Labs):**
                - `black-forest-labs/flux.2-pro` / `flux-1-dev` / `flux-1-schnell`
                *(Note: FLUX on OpenRouter is text-to-image art generation).*

                **👁️ Qwen Multimodal Models:**
                - `qwen/qwen-2.5-vl-72b-instruct` / `qwen3-vl-8b-instruct`
                *(Note: Qwen VL is vision-to-text; it analyzes documents and outputs text, not images).*
                """
            )

        with st.sidebar.expander("✏️ View / Customize AI Prompt"):
            ai_prompt = st.text_area(
                "Prompt Instructions",
                value="Remove all handwritten pen answers from this image and return the clean questions only.",
                height=100,
                help="You can customize instructions here."
            )
        default_ai_prompt = ai_prompt

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

    # Upload Options: Gallery or Camera
    upload_tab, camera_tab = st.tabs(["📁 Choose Photos (Gallery)", "📸 Take Photo with Camera"])

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Select Worksheet Photos",
            type=['jpg', 'jpeg', 'png', 'webp', 'heic', 'heif'],
            accept_multiple_files=True,
            key="mobile_file_uploader",
            help="Select photos from your phone gallery. Supports JPG, PNG, WEBP, and iPhone HEIC."
        )

    with camera_tab:
        camera_file = st.camera_input("Take a photo of a worksheet page", key="mobile_camera_input")

    # Aggregate images from both sources
    all_uploads = []
    if uploaded_files:
        for f in uploaded_files:
            all_uploads.append((f.name, f.getvalue()))
    if camera_file:
        all_uploads.append(("camera_photo.jpg", camera_file.getvalue()))

    if all_uploads:
        st.success(f"✅ **{len(all_uploads)} page(s) ready for cleaning!**")

        st.markdown("##### 📸 Uploaded Pages Preview:")
        preview_cols = st.columns(min(len(all_uploads), 3) if len(all_uploads) > 1 else 1)
        for idx, (fname, raw_bytes) in enumerate(all_uploads):
            with preview_cols[idx % len(preview_cols)]:
                thumb_bytes = cleaner.normalize_image_bytes(raw_bytes, max_dim=450)
                st.image(thumb_bytes, caption=f"Page {idx+1}: {fname}", use_container_width=True)

        st.markdown("---")

        if st.button("🧹 Regenerate Clean Pages & PDF", type="primary", use_container_width=True):
            if "OpenRouter" in cleaning_engine and not api_key.strip():
                st.error("⚠️ Please enter your OpenRouter API Key in the sidebar (starts with sk-or-v1-...).")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_images = []

            for i, (fname, file_bytes) in enumerate(all_uploads):
                if "OpenRouter" in cleaning_engine:
                    status_text.text(f"🎨 Nano Banana Pro regenerating page {i+1} of {len(all_uploads)}...")
                    cleaned_bgr, msg = cleaner.regenerate_image_with_openrouter(
                        file_bytes, api_key, chosen_model=chosen_model, custom_prompt=default_ai_prompt
                    )
                else:
                    status_text.text(f"Processing page {i+1} of {len(all_uploads)}...")
                    cleaned_bgr, msg = cleaner.process_image(
                        file_bytes, mode=cv_mode, sensitivity=sensitivity, enhance=enhance
                    )

                if cleaned_bgr is None:
                    st.error(f"Page {i+1} ({fname}): {msg}")
                    continue

                if "Warning" in msg:
                    st.warning(f"Page {i+1} ({fname}): {msg}")

                processed_images.append(cleaned_bgr)

                # Show Side-by-Side Image Preview
                if show_preview:
                    st.markdown(f"**Page {i+1}: {fname}**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        orig_preview = cleaner.normalize_image_bytes(file_bytes, max_dim=800)
                        st.image(orig_preview, caption="Original Uploaded Photo", use_container_width=True)
                    with col2:
                        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
                        st.image(cleaned_rgb, caption="✨ Cleaned Output", use_container_width=True)

                progress_bar.progress((i + 1) / len(all_uploads))

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
            💡 <strong>Powered by Nano Banana Pro:</strong> Takes your solved booklet photos and directly synthesizes clean digital worksheet images with all answers erased.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
