"""
Streamlit Web Application for Worksheet Cleaner.
Mobile-first UI to upload photos, erase handwritten answers, and compile the
clean pages into a PDF. Engines:
- Detect & Erase (default): a cheap vision model finds the handwriting, OpenCV erases it.
- AI Regeneration: an image model redraws the whole page (OpenRouter).
- Local OpenCV Filter: offline color filter.
"""

import streamlit as st
import os
import cv2
import numpy as np
import handwriting_eraser
from cleaner import DEFAULT_REGEN_MODEL, WorksheetCleaner

ENGINE_DETECT = "🔍 Detect & Erase (cheap, keeps printed text exact)"
ENGINE_REGENERATE = "🍌 AI Regeneration (Nano Banana 2.1, redraws page)"
ENGINE_OPENCV = "🎨 Local OpenCV Filter (Offline)"

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


API_KEYS = {
    # provider: (secret / env var name, sidebar label, placeholder, where to get one)
    handwriting_eraser.PROVIDER_GEMINI: (
        "GEMINI_API_KEY", "Gemini API Key", "AIza...",
        "[Get a free key at aistudio.google.com/apikey](https://aistudio.google.com/apikey)",
    ),
    handwriting_eraser.PROVIDER_OPENROUTER: (
        "OPENROUTER_API_KEY", "OpenRouter API Key", "sk-or-v1-...",
        "[Get API Key at openrouter.ai/keys](https://openrouter.ai/keys)",
    ),
}


def server_api_key(name: str) -> str:
    """Key from Streamlit secrets or the environment ("" if none)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except FileNotFoundError:  # no secrets.toml (StreamlitSecretNotFoundError subclasses it)
        pass
    return os.environ.get(name, "")


def api_key_input(provider: str) -> str:
    """
    Sidebar key field. A server-side key is used directly and never put into
    the widget, which would send it to the browser.
    """
    secret_name, label, placeholder, link = API_KEYS[provider]
    server_key = server_api_key(secret_name)
    typed = st.sidebar.text_input(
        label,
        type="password",
        key=f"api_key_{provider}",
        placeholder="Using the app's configured key" if server_key else placeholder,
    )
    st.sidebar.markdown(f"👉 {link}")
    return typed.strip() or server_key


def main():
    st.title("📝 Worksheet Cleaner")
    st.markdown("Regenerate clean assignment images without answers — upload photos, get a clean printable PDF.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    cleaning_engine = st.sidebar.radio(
        "Cleaning Engine",
        [ENGINE_DETECT, ENGINE_REGENERATE, ENGINE_OPENCV],
        help=(
            "Detect & Erase: a cheap vision model (~$0.002/page) only finds the handwriting; "
            "it is erased locally, so printed text is never changed. "
            "AI Regeneration: an image model redraws the whole page (~$0.03/page with Nano Banana 2.1) and may alter printed text."
        ),
    )
    needs_api_key = cleaning_engine in (ENGINE_DETECT, ENGINE_REGENERATE)

    api_key = ""
    chosen_model = ""
    default_ai_prompt = ""

    if cleaning_engine == ENGINE_DETECT:
        provider_labels = {
            "Gemini API (free tier)": handwriting_eraser.PROVIDER_GEMINI,
            "OpenRouter (paid, no geo-blocks)": handwriting_eraser.PROVIDER_OPENROUTER,
        }
        detector_provider = provider_labels[st.sidebar.radio(
            "Detection Provider",
            list(provider_labels.keys()),
            help=(
                "Same Gemini models either way. The Gemini API is free within its rate limits; "
                "on the free tier Google may use requests to improve its products. "
                "OpenRouter costs ~$0.002/page and works where the Gemini API is unavailable."
            ),
        )]
        api_key = api_key_input(detector_provider)
        provider_name = "Gemini" if detector_provider == handwriting_eraser.PROVIDER_GEMINI else "OpenRouter"
        detector_options = {
            "Gemini 3.5 Flash-Lite [Recommended, most accurate boxes]": "google/gemini-3.5-flash-lite",
            "Gemini 3.1 Flash-Lite [Cheapest, looser boxes]": "google/gemini-3.1-flash-lite",
            "✏️ Custom Model ID...": "custom",
        }
        detector_label = st.sidebar.selectbox("Detection Model", options=list(detector_options.keys()))
        detector_model = detector_options[detector_label]
        if detector_model == "custom":
            detector_model = st.sidebar.text_input(
                "Enter Model ID", value=handwriting_eraser.DEFAULT_DETECTOR_MODEL,
                help="With or without the 'google/' prefix."
            )
        scan_look = st.sidebar.toggle(
            "Scanned look (white paper)", value=True,
            help="Evens out lighting so the paper prints white. Turn off to keep the photo's paper color."
        )
        show_boxes = st.sidebar.toggle(
            "Show detected handwriting boxes", value=True,
            help="Green = written answer, orange = circle/underline over printed text."
        )

    elif cleaning_engine == ENGINE_REGENERATE:
        api_key = api_key_input(handwriting_eraser.PROVIDER_OPENROUTER)
        provider_name = "OpenRouter"
        model_options = {
            "⚡ Nano Banana 2.1 (google/gemini-nano-banana-2.1) [Recommended, ~4x cheaper]": "google/gemini-nano-banana-2.1",
            "🍌 Nano Banana Pro (google/gemini-3-pro-image) [Highest quality, ~$0.13/page]": "google/gemini-3-pro-image",
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
            help="Select the AI model. For worksheets, use Image-to-Image models (Nano Banana 2.1 / Pro / Gemini / GPT-5 Image)."
        )

        if model_options[selected_label] == "custom":
            chosen_model = st.sidebar.text_input("Enter OpenRouter Model Slug", value=DEFAULT_REGEN_MODEL)
        else:
            chosen_model = model_options[selected_label]

        with st.sidebar.expander("ℹ️ Model Info & Modalities"):
            st.markdown(
                """
                **🍌 Google Image Models (Native Image-to-Image):**
                - `google/gemini-nano-banana-2.1` (Nano Banana 2.1 — default, fast & ~4x cheaper than Pro)
                - `google/gemini-3-pro-image` (Nano Banana Pro — Highest quality text & question preservation)
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
            if needs_api_key and not api_key:
                st.error(f"⚠️ Please enter your {provider_name} API Key in the sidebar.")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_images = []

            for i, (fname, file_bytes) in enumerate(all_uploads):
                marks = []
                if cleaning_engine == ENGINE_DETECT:
                    status_text.text(f"🔍 Finding handwriting on page {i+1} of {len(all_uploads)}...")
                    page_bytes = cleaner.normalize_image_bytes(file_bytes)
                    cleaned_bgr, marks, msg = handwriting_eraser.detect_and_erase(
                        page_bytes, api_key, model=detector_model, scan_look=scan_look,
                        provider=detector_provider,
                    )
                elif cleaning_engine == ENGINE_REGENERATE:
                    status_text.text(f"🎨 Regenerating page {i+1} of {len(all_uploads)}...")
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
                        if marks and show_boxes:
                            overlay = handwriting_eraser.draw_marks(
                                cv2.imdecode(np.frombuffer(page_bytes, np.uint8), cv2.IMREAD_COLOR), marks
                            )
                            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                     caption="Detected handwriting", use_container_width=True)
                        else:
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
            💡 <strong>Detect &amp; Erase:</strong> a vision model finds the handwritten answers and they are erased locally, so the printed questions stay exactly as in your photo.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
