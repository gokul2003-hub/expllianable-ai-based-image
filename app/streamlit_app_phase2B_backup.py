"""
Streamlit Demonstration Application for Vision Foundation Model Authenticity Verification.

Project Title:
Explainable AI-Based Image Authenticity Verification and AI Edit Detection using Vision Foundation Models

Pipeline:
Input Image -> DINOv2-Small (ViT-S/14) -> 384-D Representation -> MLP Head
-> Calibrated Confidence -> Predictive Entropy -> Gradient Patch Attribution
-> Representation OOD Indicator -> Optional TTA -> Frequency Diagnostics
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st

from app.models_vfm import (
    load_dinov2_small,
    get_dinov2_transforms,
    DINOv2ClassificationHead,
    DINOv2AuthenticityClassifier,
    EMBEDDING_DIM,
    DEFAULT_IMAGE_SIZE
)
from app.explainability import DINOv2PatchAttributor
from app.calibration import compute_entropy
from app.frequency_features import extract_compact_fft_features


# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Authenticity Verification (DINOv2)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── App background ── */
.stApp { background: #f0f2f6; }
.block-container { padding-top: 1.5rem !important; max-width: 1280px; }

/* ── Dark navy sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a 0%,#1e293b 100%) !important;
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #f1f5f9 !important;
    font-size: 0.84rem;
    font-weight: 500;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] hr { border-color: #475569; }
section[data-testid="stSidebar"] .stCaption { color: #94a3b8 !important; font-size: 0.72rem !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #1e293b !important;
    border-color: #475569 !important;
}
section[data-testid="stSidebar"] p { color: #e2e8f0 !important; }

/* ── Section header ── */
.sec-label {
    font-size: 0.6rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: #64748b; margin-bottom: 0.1rem;
}
.sec-heading {
    font-size: 1.15rem; font-weight: 700; color: #0f172a;
    margin: 0 0 0.15rem; line-height: 1.2;
}
.sec-sub {
    font-size: 0.72rem; color: #64748b; margin-bottom: 0.8rem; line-height: 1.5;
}

/* ── White info card (metadata, OOD) ── */
.info-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 1rem 1.1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); height: 100%;
}
.info-card-label {
    font-size: 0.58rem; font-weight: 800; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748b; margin-bottom: 0.6rem;
}

/* ── Evidence cards (A / B / C) ── */
.ev-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.85rem 0.85rem 0.75rem;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 0.5rem;
    min-height: 5.5rem;
}
.ev-letter {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.7rem; height: 1.7rem;
    border-radius: 50%;
    background: #3b82f6;
    color: #ffffff;
    font-weight: 800; font-size: 0.85rem;
    margin-bottom: 0.45rem;
}
.ev-title {
    font-size: 0.82rem; font-weight: 700; color: #0f172a;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.12rem;
}
.ev-sub  { font-size: 0.68rem; color: #64748b; margin-bottom: 0.45rem; }
.ev-cap  {
    font-size: 0.67rem; color: #94a3b8; margin-top: 0.55rem;
    line-height: 1.55; padding: 0 0.05rem;
}

/* ── Attribution legend bar ── */
.attr-legend {
    display: flex; align-items: center; gap: 0.4rem;
    margin-top: 0.55rem; font-size: 0.65rem; color: #64748b;
}
.al-bar {
    flex: 1; height: 8px; border-radius: 4px;
    background: linear-gradient(to right,#312e81,#3730a3,#2563eb,#06b6d4,#fbbf24,#ef4444);
}

/* ── Image resolution note ── */
.res-note {
    font-size: 0.62rem; color: #92400e; background: #fffbeb;
    border: 1px solid #fde68a; border-radius: 5px;
    padding: 0.25rem 0.55rem; margin-top: 0.3rem;
    line-height: 1.45; display: block;
}

/* ── Probability bar rows ── */
.prob-row {
    display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.45rem;
}
.prob-label {
    font-size: 0.72rem; font-weight: 600; color: #374151;
    min-width: 6.5rem; white-space: nowrap;
}
.prob-track {
    flex: 1; background: #e2e8f0; border-radius: 4px; height: 11px; overflow: hidden;
}
.prob-fill-auth  { height: 100%; border-radius: 4px; background: #16a34a; }
.prob-fill-synth { height: 100%; border-radius: 4px; background: #dc2626; }
.prob-val {
    font-size: 0.72rem; font-weight: 700; color: #0f172a;
    min-width: 3.8rem; text-align: right;
}

/* ── OOD gauge ── */
.ood-track {
    position: relative; height: 10px; border-radius: 5px;
    background: linear-gradient(to right, #22c55e 0%, #eab308 55%, #ef4444 100%);
    margin: 0.4rem 0 0.25rem;
}
.ood-needle {
    position: absolute; top: -3px;
    width: 3px; height: 16px; border-radius: 2px;
    background: #0f172a; transform: translateX(-50%);
}
.ood-thresh-line {
    position: absolute; top: 0;
    width: 2px; height: 10px;
    background: rgba(255,255,255,0.8);
}

/* ── Divider ── */
.sec-divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.2rem 0; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def H(html: str):
    st.markdown(html, unsafe_allow_html=True)


def hr():
    H('<hr class="sec-divider">')


def section(label: str, heading: str, subtitle: str = ""):
    H(f'<div class="sec-label">{label}</div>'
      f'<div class="sec-heading">{heading}</div>'
      + (f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""))


def _display_img(pil_img: Image.Image, max_px: int = 512) -> Image.Image:
    """Return a display-only upscaled copy using LANCZOS for low-res sources.
    The original PIL image and all inference preprocessing are UNCHANGED.
    Renders at 512px so the browser scales DOWN to fit the column (avoids upscaling blur).
    """
    w, h = pil_img.width, pil_img.height
    if w < max_px and h < max_px:
        scale = max_px // max(w, h)
        if scale > 1:
            return pil_img.resize((w * scale, h * scale), Image.NEAREST)
    return pil_img


# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_verification_pipeline():
    """Caches pipeline resources for fast subsequent inference."""
    device = torch.device("cpu")

    # 1. Explainability & Classification Attributor
    attributor = DINOv2PatchAttributor(
        checkpoint_path="models/dinov2_authenticity_head.pth",
        device=device
    )

    # 2. Calibration Parameters
    calib_path = "models/calibration_params.json"
    temp = 1.0
    if os.path.exists(calib_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            cdata = json.load(f)
            temp = float(cdata.get("temperature", 1.0))

    # 3. OOD Parameters
    ood_path = "models/ood_params.json"
    ood_params = None
    if os.path.exists(ood_path):
        with open(ood_path, "r", encoding="utf-8") as f:
            ood_params = json.load(f)

    return attributor, temp, ood_params, device


def compute_ood_score(features_np: np.ndarray, ood_params: dict):
    """Computes minimum cosine distance to training class centroids."""
    if ood_params is None:
        return 0.0, 0.6673, False

    mu_auth  = np.array(ood_params["centroid_authentic"],  dtype=np.float32)
    mu_synth = np.array(ood_params["centroid_synthetic"],  dtype=np.float32)

    feat_norm  = np.linalg.norm(features_np) + 1e-12
    cos_auth   = np.dot(features_np, mu_auth)  / (feat_norm * (np.linalg.norm(mu_auth)  + 1e-12))
    cos_synth  = np.dot(features_np, mu_synth) / (feat_norm * (np.linalg.norm(mu_synth) + 1e-12))

    dist_auth  = 1.0 - float(cos_auth)
    dist_synth = 1.0 - float(cos_synth)
    min_dist   = min(dist_auth, dist_synth)

    threshold = float(ood_params.get("threshold_95", 0.6673))
    is_ood    = (min_dist > threshold)
    return min_dist, threshold, is_ood


# ─────────────────────────────────────────────────────────────────────────────
def main():

    attributor, temperature, ood_params, device = load_verification_pipeline()

    # ── Hero header ──────────────────────────────────────────────────────────
    H("""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#1e40af 100%);
                border-radius:12px;padding:1.6rem 2rem 1.4rem;margin-bottom:1.2rem;
                box-shadow:0 4px 24px rgba(0,0,0,0.18)">
      <div style="font-size:0.6rem;font-weight:800;letter-spacing:0.18em;
                  text-transform:uppercase;color:#93c5fd;margin-bottom:0.35rem">
        Explainable AI &mdash; Final Year Capstone
      </div>
      <div style="font-size:1.55rem;font-weight:800;color:#f8fafc;line-height:1.2;margin-bottom:0.3rem">
        🛡️ Image Authenticity Verification &amp; AI Edit Detection
      </div>
      <div style="font-size:0.78rem;color:#94a3b8;line-height:1.5">
        Vision Foundation Model &nbsp;&bull;&nbsp; DINOv2-Small (ViT-S/14) &nbsp;&bull;&nbsp;
        Calibrated MLP Head &nbsp;&bull;&nbsp; Gradient Attribution &nbsp;&bull;&nbsp;
        OOD Diagnostics &nbsp;&bull;&nbsp; Frequency Analysis
      </div>
    </div>
    """)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="font-size:0.62rem;font-weight:800;letter-spacing:0.14em;'
        'text-transform:uppercase;color:#7dd3fc;padding:0.5rem 0 0.3rem">'
        '📁 &nbsp;Image Source</div>',
        unsafe_allow_html=True,
    )
    source_choice = st.sidebar.radio(
        "Input Source",
        ["Upload Custom Image", "Benchmark Test Preset"],
        label_visibility="collapsed"
    )

    img_input = None

    if source_choice == "Upload Custom Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload an Image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            img_input = Image.open(uploaded_file).convert("RGB")
    else:
        preset_options = {
            "Authentic — Test Set":                    os.path.join("dataset", "test", "authentic", "0005 (8).jpg"),
            "AI-Generated — Test Set":                 os.path.join("dataset", "test", "ai_edited", "1011 (4).jpg"),
            "Borderline — Authentic (misclassified)":  os.path.join("dataset", "test", "authentic", "0064 (3).jpg"),
            "Borderline — Synthetic (misclassified)":  os.path.join("dataset", "test", "ai_edited", "1040.jpg"),
        }
        selected_preset = st.sidebar.selectbox(
            "Select Test Image", list(preset_options.keys())
        )
        preset_path = preset_options[selected_preset]
        if os.path.exists(preset_path):
            img_input = Image.open(preset_path).convert("RGB")
            st.sidebar.caption(f"📂 `{preset_path}`")
        else:
            st.sidebar.error("Preset file not found.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="font-size:0.62rem;font-weight:800;letter-spacing:0.14em;'
        'text-transform:uppercase;color:#7dd3fc;padding:0.2rem 0 0.3rem">'
        '⚙️ &nbsp;Analysis Options</div>',
        unsafe_allow_html=True,
    )
    enable_tta     = st.sidebar.checkbox(
        "Enable Test-Time Augmentation (TTA)", value=False,
        help="Averages predictions of original and horizontally flipped views. ~2× CPU latency."
    )
    show_frequency = st.sidebar.checkbox(
        "Show Frequency-Domain Diagnostic (FFT)", value=True,
        help="Supplementary 2D FFT spectrum. Does NOT influence the verdict."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="font-size:0.62rem;font-weight:800;letter-spacing:0.14em;'
        'text-transform:uppercase;color:#7dd3fc;padding:0.2rem 0 0.2rem">'
        'ℹ️ &nbsp;System</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Backbone: `facebook/dinov2-small`")
    st.sidebar.caption(f"Calibration T = `{temperature:.4f}`")
    st.sidebar.caption("Hardware: CPU inference")

    # ── Empty state ───────────────────────────────────────────────────────────
    if img_input is None:
        H("""
        <div style="background:#fff;border:1.5px dashed #cbd5e1;border-radius:12px;
                    padding:3rem 2rem;text-align:center;color:#64748b;margin-top:1.5rem">
          <div style="font-size:2.5rem;margin-bottom:0.6rem">📂</div>
          <div style="font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:0.3rem">
            No Image Selected
          </div>
          <div style="font-size:0.78rem;color:#64748b;line-height:1.6">
            Upload a custom image or select a benchmark preset from the sidebar<br>
            to begin DINOv2 authenticity analysis.
          </div>
        </div>
        """)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # ── 01 — INPUT IMAGE ─────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    hr()
    orig_w, orig_h = img_input.width, img_input.height
    is_benchmark   = (orig_w <= 64 and orig_h <= 64)

    section(
        "01 — Input",
        "Input Image",
        f"Source image: {orig_w}×{orig_h} px "
        + ("(CIFAKE benchmark — 32×32 px native resolution, not a high-resolution image)."
           if is_benchmark else "(custom upload).")
    )

    # Display-only upscaled copy for visual clarity — LANCZOS interpolation smooths low-res sources
    disp_img = _display_img(img_input, max_px=512)
    # Model input preview (224×224) — separate, used only for display
    disp_224 = img_input.resize((224, 224), Image.LANCZOS)

    in1, in2, in3, in4 = st.columns([1, 0.15, 1, 1.8], gap="small")

    with in1:
        st.image(disp_img, use_container_width=True)
        H(f'<p class="ev-cap" style="text-align:center">'
          f'<b>Original Source</b><br>{orig_w}&times;{orig_h} px'
          + (' &nbsp;&mdash;&nbsp; <span style="color:#f59e0b;font-weight:700">'
             'CIFAKE benchmark</span>' if is_benchmark else '')
          + '</p>')
        if is_benchmark:
            H('<span class="res-note">'
              'Native source resolution: {orig_w}&times;{orig_h} px. '
              'Display enlargement cannot recover missing source detail.'
              '</span>')

    with in2:
        H('<div style="display:flex;align-items:center;justify-content:center;'
          'height:100%;font-size:1.4rem;color:#94a3b8">&rarr;</div>')

    with in3:
        st.image(disp_224, use_container_width=True)
        H('<p class="ev-cap" style="text-align:center">'
          '<b>Model Input Preview</b><br>224&times;224 px &mdash; DINOv2 preprocessing</p>')
        H('<span class="res-note">'
          '📌 Display preview only. Inference uses <code>attributor.transform()</code> '
          'internally &mdash; this image does not pass through the model.'
          '</span>')

    with in4:
        H(f"""
        <div class="info-card">
          <div class="info-card-label">Image Metadata</div>
          <div style="font-size:0.78rem;color:#0f172a;line-height:2.1">
            <b>Source Resolution:</b> {orig_w}&times;{orig_h} px<br>
            <b>Colour Mode:</b> RGB<br>
            <b>Backbone Input:</b> 224&times;224 px<br>
            <b>Backbone:</b> <code>facebook/dinov2-small</code><br>
            <b>Architecture:</b> ViT-S/14 (frozen, 21M params)<br>
            <b>Calibration T:</b> {temperature:.4f}<br>
            <b>Patch Grid:</b> 16&times;16 (256 tokens)<br>
            <b>Feature Dim:</b> 384-D CLS token
          </div>
        </div>
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # ── INFERENCE (ALL ML COMPUTATION — UNCHANGED) ───────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    with st.spinner("Running DINOv2 inference and gradient attribution..."):
        t0 = time.perf_counter()

        # Step A: Gradient Patch Attribution & Prediction
        explanation = attributor.explain(img_input)
        total_time  = (time.perf_counter() - t0) * 1000

        # Step B: Calibrated Probabilities
        input_tensor = attributor.transform(img_input).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_feat   = attributor.backbone(input_tensor)   # [1, 384]
            raw_logits = attributor.head(cls_feat)            # [1, 2]
            cal_logits = raw_logits / temperature
            cal_probs  = F.softmax(cal_logits, dim=-1).squeeze(0).cpu().numpy()

        classes   = attributor.classes
        idx_auth  = classes.index("authentic")
        idx_synth = classes.index("ai_edited")

        prob_auth  = float(cal_probs[idx_auth])
        prob_synth = float(cal_probs[idx_synth])

        # Binary Decision
        if prob_synth >= 0.50:
            pred_label   = "AI-Generated / Synthetic"
            pred_conf    = prob_synth
            is_synthetic = True
        else:
            pred_label   = "Authentic"
            pred_conf    = prob_auth
            is_synthetic = False

        entropy_val = float(compute_entropy(np.array([[prob_synth, prob_auth]]))[0])

        # Step C: OOD Cosine Distance
        cls_feat_np = cls_feat.squeeze(0).cpu().numpy()
        ood_dist, ood_thresh, is_ood = compute_ood_score(cls_feat_np, ood_params)

        # Step D: Optional TTA  (0.50 × original + 0.50 × horizontal flip — UNCHANGED)
        tta_executed   = False
        tta_prob_auth  = None
        tta_prob_synth = None
        if enable_tta:
            img_flipped    = img_input.transpose(Image.FLIP_LEFT_RIGHT)
            flip_tensor    = attributor.transform(img_flipped).unsqueeze(0).to(device)
            with torch.no_grad():
                cls_flip        = attributor.backbone(flip_tensor)
                flip_logits     = attributor.head(cls_flip)
                cal_flip_logits = flip_logits / temperature
                flip_probs      = F.softmax(cal_flip_logits, dim=-1).squeeze(0).cpu().numpy()
            tta_prob_auth  = 0.5 * (prob_auth  + float(flip_probs[idx_auth]))
            tta_prob_synth = 0.5 * (prob_synth + float(flip_probs[idx_synth]))
            tta_executed   = True

    # ─────────────────────────────────────────────────────────────────────────
    # ── 02 — VERDICT ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    hr()
    section("02 — Classification", "Authenticity Verdict",
            "Primary DINOv2 classification result. Supplementary diagnostics are in sections 03–05.")

    verdict_bg   = "linear-gradient(135deg,#dc2626,#991b1b)" if is_synthetic else "linear-gradient(135deg,#16a34a,#15803d)"
    verdict_icon = "🔴" if is_synthetic else "🟢"
    H(f"""
    <div style="background:{verdict_bg};border-radius:10px;padding:1.1rem 1.5rem;
                color:#fff;margin-bottom:0.9rem;box-shadow:0 4px 16px rgba(0,0,0,0.15)">
      <div style="font-size:0.58rem;font-weight:800;letter-spacing:0.14em;
                  text-transform:uppercase;opacity:0.75;margin-bottom:0.25rem">
        Classification Verdict
      </div>
      <div style="font-size:1.4rem;font-weight:800;line-height:1.2;margin-bottom:0.2rem">
        {verdict_icon} &nbsp;{pred_label}
      </div>
      <div style="font-size:0.75rem;opacity:0.88;display:flex;gap:1.2rem;flex-wrap:wrap">
        <span>Calibrated confidence: <b>{pred_conf*100:.2f}%</b></span>
        <span>Entropy: <b>{entropy_val:.3f} bits</b></span>
        <span>Latency: <b>{total_time:.1f} ms</b></span>
      </div>
    </div>
    """)

    # 4 metric tiles
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Calibrated Confidence", f"{pred_conf*100:.2f}%",
                  help="Post-hoc temperature-scaled probability for the predicted class (T=0.9464).")
    with m2:
        st.metric("Predictive Entropy", f"{entropy_val:.3f} bits",
                  help="Shannon entropy of class probabilities. <0.3 bits = certain; >0.7 bits = ambiguous.")
    with m3:
        st.metric("P(Synthetic)", f"{prob_synth*100:.2f}%",
                  help="Calibrated probability that the image is AI-generated. Decision threshold: 0.50.")
    with m4:
        st.metric("Inference Latency", f"{total_time:.1f} ms",
                  help="Full forward pass + gradient attribution on CPU.")

    # Custom probability bars
    auth_pct  = int(prob_auth  * 100)
    synth_pct = int(prob_synth * 100)
    H(f"""
    <div style="margin:0.6rem 0 0.2rem">
      <div class="prob-row">
        <span class="prob-label">P(Authentic)</span>
        <div class="prob-track">
          <div class="prob-fill-auth" style="width:{auth_pct}%"></div>
        </div>
        <span class="prob-val">{prob_auth*100:.2f}%</span>
      </div>
      <div class="prob-row">
        <span class="prob-label">P(Synthetic)</span>
        <div class="prob-track">
          <div class="prob-fill-synth" style="width:{synth_pct}%"></div>
        </div>
        <span class="prob-val">{prob_synth*100:.2f}%</span>
      </div>
    </div>
    """)

    # Confidence & Uncertainty expander
    with st.expander("📊 Confidence & Uncertainty — Interpretation", expanded=True):
        cu_c1, cu_c2, cu_c3, cu_c4 = st.columns(4)
        with cu_c1:
            st.metric("Calibrated Confidence", f"{pred_conf*100:.2f}%",
                      help="Post-hoc temperature-scaled probability assigned to the predicted class.")
        with cu_c2:
            st.metric("Predictive Entropy", f"{entropy_val:.3f} bits",
                      help="Shannon entropy of calibrated class probabilities. Low = certain, High = uncertain.")
        with cu_c3:
            st.metric("P(Authentic)", f"{prob_auth*100:.4f}%")
        with cu_c4:
            st.metric("P(Synthetic)", f"{prob_synth*100:.4f}%")

        conf_pct = pred_conf * 100
        if conf_pct >= 90 and entropy_val < 0.3:
            st.success(
                f"**High confidence ({conf_pct:.1f}%):** The calibrated model probability is strongly "
                f"concentrated on the '{pred_label}' class. "
                f"Low predictive entropy ({entropy_val:.3f} bits) indicates a stable, well-separated decision."
            )
        elif conf_pct >= 70 and entropy_val < 0.7:
            st.info(
                f"**Moderate confidence ({conf_pct:.1f}%):** The model leans toward '{pred_label}' "
                f"but predictive entropy ({entropy_val:.3f} bits) is non-trivial. "
                "Consider supplementary diagnostics (OOD, TTA, FFT) before drawing conclusions."
            )
        else:
            st.warning(
                f"**Low confidence ({conf_pct:.1f}%):** Probabilities are close to the 0.50 decision boundary. "
                f"Predictive entropy ({entropy_val:.3f} bits) is elevated — borderline case. "
                "Do not treat the verdict as reliable without additional evidence."
            )
        st.caption(
            "ℹ️ *Confidence = calibrated probability concentrated on one class. "
            "Entropy = uncertainty. Higher entropy → less certain. "
            "Confidence alone is not forensic proof.*"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ── 03 — EXPLAINABILITY — VISUAL EVIDENCE ────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    hr()
    section("03 — Explainability", "Visual Evidence",
            "How the model spatially attributed its authenticity decision.")

    # Render presentation copies at 512x512 for sharp display (browser scales DOWN, not up)
    # These are display-only and are NEVER fed back into the model
    img_224_display = img_input.resize((512, 512), Image.LANCZOS)
    heatmap_display = Image.fromarray(explanation["colored_heatmap"]).resize((512, 512), Image.LANCZOS)
    overlay_display = explanation["overlay_image"].resize((512, 512), Image.LANCZOS)

    ev1, ev2, ev3 = st.columns(3, gap="medium")

    with ev1:
        H('<div class="ev-card">'
          '<div><span class="ev-letter">A</span></div>'
          '<div class="ev-title">Source / Model Input</div>'
          '<div class="ev-sub">224 &times; 224 px &mdash; DINOv2 preprocessing</div>'
          '</div>')
        st.image(img_224_display, use_container_width=True)
        H('<p class="ev-cap">'
          'Standardized input tensor as received by the DINOv2 backbone. '
          'Resizing changes scale — it does <em>not</em> add new image detail. '
          'Displayed with LANCZOS interpolation for visual smoothness only; '
          'the model receives its own internal transform.</p>')

    with ev2:
        H('<div class="ev-card">'
          '<div><span class="ev-letter">B</span></div>'
          '<div class="ev-title">Patch Attribution Heatmap</div>'
          '<div class="ev-sub">16&times;16 gradient grid &rarr; 224&times;224 bicubic</div>'
          '</div>')
        st.image(heatmap_display, use_container_width=True)
        H('<div class="attr-legend">'
          '<span style="font-size:0.65rem;color:#64748b;white-space:nowrap">Low&nbsp;relevance</span>'
          '<div class="al-bar"></div>'
          '<span style="font-size:0.65rem;color:#64748b;white-space:nowrap">High&nbsp;relevance</span>'
          '</div>'
          '<p class="ev-cap" style="text-align:center">'
          'Gradient-based patch relevance scores upsampled from the 16&times;16 ViT '
          'patch grid. Indicates <em>model attention</em>, not ground-truth tampering regions.</p>')

    with ev3:
        H('<div class="ev-card">'
          '<div><span class="ev-letter">C</span></div>'
          '<div class="ev-title">Attribution Overlay</div>'
          '<div class="ev-sub">Heatmap composited over model input</div>'
          '</div>')
        st.image(overlay_display, use_container_width=True)
        H('<p class="ev-cap">'
          'Heatmap overlaid on the model input image. Brighter/warmer regions received '
          'higher gradient attribution. <em>Highlighted areas do not constitute '
          'forensic evidence of manipulation.</em></p>')

    H('<p style="font-size:0.68rem;color:#94a3b8;margin-top:0.3rem">'
      '⚠️ Attribution is approximate gradient-based analysis relative to the model\'s '
      'training data (CIFAKE). It is not a pixel-level ground-truth tampering mask.</p>')

    # ─────────────────────────────────────────────────────────────────────────
    # ── 04 — DIAGNOSTICS: OOD ────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    hr()
    section("04 — Diagnostics", "Model Diagnostics",
            "Supplementary indicators. These do NOT override the primary classifier verdict.")

    ood_col1, ood_col2 = st.columns([1, 2], gap="medium")

    with ood_col1:
        # OOD visual gauge — clamp to [0, 1] for display
        gauge_max  = max(ood_thresh * 1.5, ood_dist * 1.1, 1.0)
        dist_pct   = min(ood_dist  / gauge_max * 100, 100)
        thresh_pct = min(ood_thresh / gauge_max * 100, 100)
        ood_color  = "#ef4444" if is_ood else "#16a34a"

        H(f"""
        <div class="info-card">
          <div class="info-card-label">Out-of-Distribution (OOD) Indicator</div>
          <div style="font-size:1.5rem;font-weight:800;color:{ood_color};margin-bottom:0.05rem">
            {ood_dist:.4f}
          </div>
          <div style="font-size:0.7rem;color:#64748b;margin-bottom:0.1rem">
            Cosine centroid distance
          </div>
          <div class="ood-track">
            <div class="ood-thresh-line" style="left:{thresh_pct:.1f}%"></div>
            <div class="ood-needle"      style="left:{dist_pct:.1f}%"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.6rem;color:#94a3b8">
            <span>0</span>
            <span>Threshold {ood_thresh:.4f}</span>
            <span>{gauge_max:.2f}</span>
          </div>
          <div style="font-size:0.67rem;color:#64748b;margin-top:0.4rem">
            95th-percentile threshold: <b>{ood_thresh:.4f}</b><br>
            Status: <b style="color:{ood_color}">
              {"⚠️ Atypical (OOD)" if is_ood else "✅ In-Distribution"}
            </b>
          </div>
        </div>
        """)

    with ood_col2:
        st.markdown("**What is the OOD Indicator?**")
        st.markdown(
            "The 384-D DINOv2 feature vector is compared to the training-set class centroids "
            "using cosine distance. A distance exceeding the 95th-percentile validation threshold "
            "flags the representation as atypical."
        )
        if is_ood:
            st.warning(
                f"⚠️ **Atypical Representation (OOD Flag):** Distance `{ood_dist:.4f}` "
                f"exceeds the 95th-percentile threshold `{ood_thresh:.4f}`.\n\n"
                "*An OOD flag means the image's feature representation is unusual relative to the training "
                "distribution. It does NOT automatically prove the image is synthetic or tampered.*"
            )
        else:
            st.success(
                f"✅ **In-Distribution:** Distance `{ood_dist:.4f}` is within the "
                f"95th-percentile threshold `{ood_thresh:.4f}`. "
                "The image feature representation is consistent with the training distribution."
            )

    # ── ROBUSTNESS CHECK — TEST-TIME AUGMENTATION ────────────────────────────
    if tta_executed:
        hr()
        section("04b — Robustness", "Robustness Check — Test-Time Augmentation",
                "Two-view aggregation: 0.50 × Original + 0.50 × Horizontal Flip. "
                "Robustness diagnostic only — does NOT change the primary verdict.")

        orig_verdict = "AI-Generated / Synthetic" if prob_synth >= 0.50 else "Authentic"
        tta_verdict  = "AI-Generated / Synthetic" if tta_prob_synth >= 0.50 else "Authentic"
        agree        = orig_verdict == tta_verdict
        delta        = (tta_prob_synth - prob_synth) * 100

        tta_c1, tta_c2, tta_c3, tta_c4 = st.columns(4, gap="small")

        with tta_c1:
            H(f"""
            <div class="info-card">
              <div class="info-card-label">Single-Crop (Original)</div>
              <div style="font-size:0.8rem;font-weight:700;color:#0f172a;margin-bottom:0.4rem">
                {orig_verdict}
              </div>
              <div style="font-size:0.72rem;color:#64748b;line-height:1.8">
                P(Auth): <b>{prob_auth*100:.2f}%</b><br>
                P(Synth): <b>{prob_synth*100:.2f}%</b>
              </div>
            </div>
            """)

        with tta_c2:
            H(f"""
            <div class="info-card">
              <div class="info-card-label">TTA — Averaged Views</div>
              <div style="font-size:0.8rem;font-weight:700;color:#0f172a;margin-bottom:0.4rem">
                {tta_verdict}
              </div>
              <div style="font-size:0.72rem;color:#64748b;line-height:1.8">
                P(Auth): <b>{tta_prob_auth*100:.2f}%</b><br>
                P(Synth): <b>{tta_prob_synth*100:.2f}%</b>
              </div>
            </div>
            """)

        with tta_c3:
            agree_color = "#16a34a" if agree else "#dc2626"
            agree_text  = f"✅ Agreement<br>{tta_verdict}" if agree else "⚠️ Disagreement<br>Views differ"
            H(f"""
            <div class="info-card" style="border-color:{agree_color}40">
              <div class="info-card-label">Prediction Agreement</div>
              <div style="font-size:0.82rem;font-weight:700;color:{agree_color};line-height:1.4">
                {agree_text}
              </div>
            </div>
            """)

        with tta_c4:
            shift_color = "#dc2626" if delta > 3 else ("#eab308" if abs(delta) > 1 else "#16a34a")
            H(f"""
            <div class="info-card">
              <div class="info-card-label">Probability Shift (Synthetic)</div>
              <div style="font-size:1.3rem;font-weight:800;color:{shift_color}">
                {delta:+.2f}%
              </div>
              <div style="font-size:0.67rem;color:#94a3b8;margin-top:0.2rem">
                TTA avg minus single-crop
              </div>
            </div>
            """)

        if agree and abs(delta) < 3.0:
            st.info(f"ℹ️ Both views agree (`{orig_verdict}`) with a small shift ({delta:+.2f}%). "
                    "Prediction is relatively stable under horizontal reflection.")
        elif agree:
            st.warning(f"ℹ️ Both views agree (`{orig_verdict}`) but show a notable shift ({delta:+.2f}%). "
                       "Some sensitivity to horizontal reflection detected.")
        else:
            st.error(f"⚠️ Views disagree (shift: {delta:+.2f}%). "
                     "High decision sensitivity near the 0.50 boundary. "
                     "Treat the result with additional caution.")

    # ─────────────────────────────────────────────────────────────────────────
    # ── 05 — FREQUENCY DIAGNOSTIC ────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    if show_frequency:
        hr()
        section("05 — Frequency", "Frequency-Domain Diagnostic Analysis",
                "Supplementary 2D FFT analysis. "
                "The DINOv2 CLS head is the primary classifier — "
                "FFT is orthogonal and does NOT influence the verdict.")

        fft_gray  = img_input.convert("L").resize((224, 224))
        fft_arr   = np.asarray(fft_gray, dtype=np.float64) / 255.0
        fft_2d    = np.fft.fftshift(np.fft.fft2(fft_arr))
        log_mag   = np.log1p(np.abs(fft_2d))
        norm_spec = ((log_mag - log_mag.min()) /
                     (log_mag.max() - log_mag.min() + 1e-12) * 255).astype(np.uint8)

        fft_feats = extract_compact_fft_features(img_input)
        radial_8  = fft_feats[:8]

        f_col1, f_col2 = st.columns([1, 1], gap="medium")
        with f_col1:
            # Render 512x512 display copy for sharp presentation (FFT calculation unchanged)
            fft_display = Image.fromarray(norm_spec).resize((512, 512), Image.LANCZOS)
            st.image(
                fft_display,
                caption="2D FFT Log-Magnitude Spectrum (DC Centred, Grayscale)",
                use_container_width=True
            )
            H('<p class="ev-cap">'
              'Spatial frequency distribution. Natural images typically show '
              'a smooth 1/f roll-off; AI-generated images may exhibit unusual spectral '
              'patterns. Diagnostic only.</p>')
        with f_col2:
            fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
            fig.patch.set_facecolor("white")
            colors  = ["#3b82f6"] * 4 + ["#f59e0b"] * 4
            ax.bar(range(1, 9), radial_8 * 100, color=colors, alpha=0.88, width=0.7)
            ax.set_xlabel("Radial Frequency Band (1 = Low / DC, 8 = Nyquist High)", fontsize=11)
            ax.set_ylabel("Energy Proportion (%)", fontsize=11)
            ax.set_title("Radial Spectral Energy Partition — 8 Bands", fontsize=12, fontweight="bold")
            ax.set_xticks(range(1, 9))
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, alpha=0.25, axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            H('<p class="ev-cap">'
              'Blue bars = low-frequency bands (dominant in most images). '
              'Amber bars = high-frequency bands. '
              'Energy distribution alone is insufficient for classification.</p>')

    # ─────────────────────────────────────────────────────────────────────────
    # ── 06 — TECHNICAL DETAILS ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    hr()
    with st.expander("06 — Technical Details & Specifications", expanded=False):
        st.markdown(f"""
| Property | Value |
|---|---|
| **Backbone** | DINOv2-Small (`facebook/dinov2-small`) |
| **Architecture** | Vision Transformer ViT-S/14 |
| **Backbone Params** | ~21M (frozen) |
| **Input Resolution** | 224 × 224 px (ImageNet normalisation) |
| **Feature Dimension** | 384-D CLS token |
| **Patch Grid** | 16 × 16 (256 spatial tokens) |
| **Head** | Linear(384→256)→ReLU→Dropout(0.1)→Linear(256→2) |
| **Calibration** | Temperature Scaling (T = {temperature:.4f}) |
| **Attribution** | Gradient w.r.t. patch tokens, bicubic upsample |
| **OOD Method** | Min cosine centroid distance, 95th-pct threshold |
| **Frequency** | 2D FFT log-magnitude, 32-D radial energy features |
| **Hardware** | CPU |
""")

    with st.expander("⚠️ Known Model Limitations", expanded=False):
        st.markdown("""
1. **Dataset Scope:** Trained on CIFAKE (Stable Diffusion v1.4 vs CIFAR-10 natural imagery). Generalisation to commercial generators (Midjourney v6, DALL-E 3, Flux) is not guaranteed.
2. **No Ground-Truth Masks:** Gradient attribution shows model attention, not human-annotated tampered pixels.
3. **Degradation Sensitivity:** Extreme blur (r=3.0 → −26.6% accuracy) and lossy downscaling (0.25× → −26.0%) significantly impair detection.
4. **OOD Benchmark:** No external OOD benchmark used; verification relies on proxy manifold testing.
5. **TTA Overhead:** ~2.01× CPU latency for +0.40% accuracy gain.
""")

    hr()
    st.warning(
        "⚠️ **Research Disclaimer:** This system provides a probabilistic AI-based authenticity assessment, "
        "not definitive forensic proof. Attribution visualizations represent approximate model attention "
        "relative to the CIFAKE training distribution. "
        "OOD detection has not been validated on an external real-world benchmark."
    )


if __name__ == "__main__":
    main()
