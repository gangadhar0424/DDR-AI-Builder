"""
DDR-AI-Builder — Streamlit Web Interface.

Provides a professional upload-and-generate UI for creating
Detailed Diagnostic Reports from inspection and thermal PDFs.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import config
from pipeline import DDRPipeline, PipelineResult


# ═══════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="DDR-AI-Builder",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e40af, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        padding: 10px 0;
        letter-spacing: -1px;
    }
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 32px;
    }

    /* Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
    }

    /* Stats */
    .stat-container {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin: 16px 0;
    }
    .stat-box {
        flex: 1;
        min-width: 140px;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Upload area */
    .upload-zone {
        border: 2px dashed rgba(99, 102, 241, 0.4);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s;
    }
    .upload-zone:hover {
        border-color: rgba(99, 102, 241, 0.8);
        background: rgba(99, 102, 241, 0.05);
    }

    /* Success/Error banners */
    .success-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }
    .error-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }

    /* Button override */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        transition: all 0.3s !important;
        width: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 24px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════
st.markdown('<h1 class="main-header">🏗️ DDR-AI-Builder</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-powered Detailed Diagnostic Reports from Inspection & Thermal PDFs</p>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════
# Sidebar — Configuration
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # LLM Provider
    provider_list = ["gemini", "openai", "anthropic"]
    default_idx = provider_list.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in provider_list else 0
    llm_provider = st.selectbox(
        "LLM Provider",
        provider_list,
        index=default_idx,
    )

    # API Key
    if llm_provider == "openai":
        api_key = st.text_input(
            "OpenAI API Key",
            value=config.OPENAI_API_KEY,
            type="password",
            help="Enter your OpenAI API key",
        )
        model = st.text_input("Model", value=config.OPENAI_MODEL)
    elif llm_provider == "anthropic":
        api_key = st.text_input(
            "Anthropic API Key",
            value=config.ANTHROPIC_API_KEY,
            type="password",
            help="Enter your Anthropic API key",
        )
        model = st.text_input("Model", value=config.ANTHROPIC_MODEL)
    else:  # gemini
        api_key = st.text_input(
            "Gemini API Key",
            value=config.GEMINI_API_KEY,
            type="password",
            help="Enter your Google Gemini API key",
        )
        model = st.text_input("Model", value=config.GEMINI_MODEL)

    st.markdown("---")
    st.markdown("### 🔧 Advanced")

    temperature = st.slider(
        "LLM Temperature", 0.0, 1.0, config.LLM_TEMPERATURE, 0.05
    )
    similarity_threshold = st.slider(
        "Similarity Threshold", 0.5, 0.95, config.SIMILARITY_THRESHOLD, 0.05,
        help="Minimum cosine similarity to merge two observations",
    )

    export_formats = st.multiselect(
        "Export Formats",
        ["html", "markdown", "pdf"],
        default=["html", "markdown"],
    )

    report_title = st.text_input(
        "Report Title (optional)",
        value="",
        placeholder="Detailed Diagnostic Report (DDR)",
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.8rem;'>"
        "DDR-AI-Builder v1.0<br>Built with ❤️ and AI</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════
# Main Content — File Upload
# ═══════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="glass-card">'
        '<h3 style="color:#60a5fa; margin-bottom:12px;">📋 Inspection Report</h3>'
        '</div>',
        unsafe_allow_html=True,
    )
    inspection_file = st.file_uploader(
        "Upload Inspection PDF",
        type=["pdf"],
        key="inspection_upload",
        label_visibility="collapsed",
    )
    if inspection_file:
        st.success(f"✅ {inspection_file.name} ({inspection_file.size / 1024:.0f} KB)")

with col2:
    st.markdown(
        '<div class="glass-card">'
        '<h3 style="color:#a78bfa; margin-bottom:12px;">🌡️ Thermal Report</h3>'
        '</div>',
        unsafe_allow_html=True,
    )
    thermal_file = st.file_uploader(
        "Upload Thermal PDF",
        type=["pdf"],
        key="thermal_upload",
        label_visibility="collapsed",
    )
    if thermal_file:
        st.success(f"✅ {thermal_file.name} ({thermal_file.size / 1024:.0f} KB)")

st.markdown("---")

# ═══════════════════════════════════════════════
# Generate Button
# ═══════════════════════════════════════════════
can_generate = inspection_file is not None and thermal_file is not None and api_key

if not api_key:
    st.warning("⚠️ Please enter your API key in the sidebar to proceed.")

if can_generate:
    generate_clicked = st.button(
        "🚀 Generate DDR Report",
        use_container_width=True,
    )
else:
    generate_clicked = st.button(
        "🚀 Generate DDR Report",
        disabled=True,
        use_container_width=True,
    )

# ═══════════════════════════════════════════════
# Pipeline Execution
# ═══════════════════════════════════════════════
if generate_clicked and can_generate:
    # Override config with sidebar values
    if llm_provider == "openai":
        config.OPENAI_API_KEY = api_key
        config.OPENAI_MODEL = model
    elif llm_provider == "anthropic":
        config.ANTHROPIC_API_KEY = api_key
        config.ANTHROPIC_MODEL = model
    else:  # gemini
        config.GEMINI_API_KEY = api_key
        config.GEMINI_MODEL = model
    config.LLM_PROVIDER = llm_provider
    config.LLM_TEMPERATURE = temperature
    config.SIMILARITY_THRESHOLD = similarity_threshold

    # Save uploaded files to temp directory
    temp_dir = Path(tempfile.mkdtemp())
    inspection_path = temp_dir / inspection_file.name
    thermal_path = temp_dir / thermal_file.name
    inspection_path.write_bytes(inspection_file.getvalue())
    thermal_path.write_bytes(thermal_file.getvalue())

    # Progress bar
    progress_bar = st.progress(0, text="Initializing pipeline...")
    status_text = st.empty()

    def update_progress(step: str, progress: float):
        progress_bar.progress(progress, text=step)
        status_text.markdown(
            f'<div style="color:#94a3b8; font-size:0.9rem;">⏳ {step}</div>',
            unsafe_allow_html=True,
        )

    # Run pipeline
    pipeline = DDRPipeline(progress_callback=update_progress)

    with st.spinner(""):
        result = pipeline.run(
            inspection_pdf=str(inspection_path),
            thermal_pdf=str(thermal_path),
            report_title=report_title or None,
            export_formats=export_formats,
        )

    status_text.empty()

    # ── Display Results ──
    if result.success:
        progress_bar.progress(1.0, text="✅ Complete!")

        st.markdown(
            '<div class="success-banner">'
            '<h3 style="color:#10b981; margin:0;">✅ DDR Report Generated Successfully!</h3>'
            f'<p style="color:#94a3b8; margin:8px 0 0 0;">Completed in {result.elapsed_seconds:.1f} seconds</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Stats
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        cols = st.columns(5)
        stats = [
            ("Total Findings", result.merge_result.total_merged if result.merge_result else 0),
            ("Corroborated", result.merge_result.corroborated_count if result.merge_result else 0),
            ("Areas", len(result.merge_result.areas) if result.merge_result else 0),
            ("Conflicts", result.conflict_report.total_conflicts if result.conflict_report else 0),
            ("Completeness", f"{result.missing_report.completeness_score:.0%}" if result.missing_report else "N/A"),
        ]
        for col, (label, value) in zip(cols, stats):
            with col:
                st.markdown(
                    f'<div class="stat-box">'
                    f'<div class="stat-number">{value}</div>'
                    f'<div class="stat-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Download buttons
        st.markdown("### 📥 Download Reports")
        dl_cols = st.columns(3)

        if result.html_path and Path(result.html_path).exists():
            with dl_cols[0]:
                html_data = Path(result.html_path).read_bytes()
                st.download_button(
                    "📄 Download HTML",
                    data=html_data,
                    file_name=Path(result.html_path).name,
                    mime="text/html",
                    use_container_width=True,
                )

        if result.markdown_path and Path(result.markdown_path).exists():
            with dl_cols[1]:
                md_data = Path(result.markdown_path).read_bytes()
                st.download_button(
                    "📝 Download Markdown",
                    data=md_data,
                    file_name=Path(result.markdown_path).name,
                    mime="text/markdown",
                    use_container_width=True,
                )

        if result.pdf_path and Path(result.pdf_path).exists():
            with dl_cols[2]:
                pdf_data = Path(result.pdf_path).read_bytes()
                st.download_button(
                    "📕 Download PDF",
                    data=pdf_data,
                    file_name=Path(result.pdf_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )

        # Preview
        if result.html_path and Path(result.html_path).exists():
            with st.expander("👁️ Preview DDR Report", expanded=False):
                html_content = Path(result.html_path).read_text(encoding="utf-8")
                st.components.v1.html(html_content, height=800, scrolling=True)

        # Detailed Results
        with st.expander("📊 Pipeline Details", expanded=False):
            st.markdown("#### Inspection Report")
            if result.inspection_parsed:
                st.write(f"- Pages: {result.inspection_parsed.total_pages}")
                st.write(f"- Headings: {len(result.inspection_parsed.all_headings)}")
                st.write(f"- Observations extracted: {len(result.inspection_observations)}")

            st.markdown("#### Thermal Report")
            if result.thermal_parsed:
                st.write(f"- Pages: {result.thermal_parsed.total_pages}")
                st.write(f"- Headings: {len(result.thermal_parsed.all_headings)}")
                st.write(f"- Observations extracted: {len(result.thermal_observations)}")

            st.markdown("#### Images")
            if result.inspection_images:
                st.write(f"- Inspection: {result.inspection_images.total_images_saved} extracted")
            if result.thermal_images:
                st.write(f"- Thermal: {result.thermal_images.total_images_saved} extracted")

            st.markdown("#### Conflicts")
            if result.conflict_report and result.conflict_report.has_conflicts():
                for c in result.conflict_report.conflicts:
                    st.warning(f"**{c.area}**: {c.description}")
            else:
                st.info("No conflicts detected between reports.")

            st.markdown("#### Missing Data")
            if result.missing_report:
                st.write(result.missing_report.summary)

    else:
        progress_bar.progress(0, text="❌ Failed")
        st.markdown(
            f'<div class="error-banner">'
            f'<h3 style="color:#ef4444; margin:0;">❌ Pipeline Failed</h3>'
            f'<p style="color:#fca5a5; margin:8px 0 0 0;">{result.error}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
