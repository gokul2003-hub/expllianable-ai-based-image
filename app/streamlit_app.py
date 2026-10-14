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

/* ── Divider (softened: short centred fade → whitespace/card separation) ── */
.sec-divider {
  border: none;
  height: 1px;
  width: 72%;
  margin: 1.7rem auto;
  background: linear-gradient(90deg, rgba(148,163,184,0) 0%, #cbd5e1 22%, #cbd5e1 78%, rgba(148,163,184,0) 100%);
  border-radius: 2px;
}

/* ── Requested: black (#000000) text — color only, targeted elements ── */
section[data-testid="stSidebar"] [data-testid^="stFileUploader"] { color: #000000 !important; }
section[data-testid="stSidebar"] [data-testid^="stFileUploader"] *:not([data-testid="stFileUploaderLabel"]) { color: #000000 !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderLabel"] { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] code { color: #000000 !important; }
section[data-testid="stSidebar"] div[style*="eff6ff"],
section[data-testid="stSidebar"] div[style*="eff6ff"] *,
section[data-testid="stSidebar"] div[style*="rgb(239, 246, 255)"],
section[data-testid="stSidebar"] div[style*="rgb(239, 246, 255)"] *,
section[data-testid="stSidebar"] div[style*="rgb(29, 78, 216)"],
section[data-testid="stSidebar"] div[style*="rgb(30, 64, 175)"] { color: #000000 !important; }
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input { color: #000000 !important; }
[role="listbox"],
[role="option"],
[role="option"] * { color: #000000 !important; }

/* ── Premium footer cards: Technical Details · Known Limitations · Disclaimer ── */
/* 1) Technical Details — clean card + polished hierarchy table */
[data-testid="stExpander"]:has(table) > details {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05), 0 12px 30px rgba(15, 23, 42, 0.06);
}
[data-testid="stExpander"]:has(table) > details > summary { padding: 1rem 1.3rem; }
[data-testid="stExpander"]:has(table) > details > summary p { font-weight: 800; letter-spacing: 0.01em; }
[data-testid="stExpander"]:has(table) table {
  width: 100%;
  border: none;
  border-collapse: collapse;
  box-shadow: none;
  margin: 0.4rem 0 0.8rem;
}
[data-testid="stExpander"]:has(table) thead th {
  background: #eef2f7;
  color: #0f172a;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: none;
  border-bottom: 2px solid #cbd5e1;
  padding: 0.75rem 1.2rem;
}
[data-testid="stExpander"]:has(table) tbody td {
  border: none;
  border-bottom: 1px solid #eef1f6;
  padding: 0.72rem 1.2rem;
  color: #1f2937;
}
[data-testid="stExpander"]:has(table) tbody tr:last-child td { border-bottom: none; }
[data-testid="stExpander"]:has(table) tbody tr:nth-child(even) td { background: #f8fafc; }
[data-testid="stExpander"]:has(table) tbody td:first-child { font-weight: 700; color: #0f172a; }

/* 2) Known Model Limitations -- white card, subtle amber accent */
[data-testid="stExpander"]:has(ol) > details {
  background: #ffffff;
  border: 1px solid #eef2f7;
  border-left: 4px solid #f59e0b;
  border-radius: 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 10px 26px rgba(15, 23, 42, 0.05);
}
[data-testid="stExpander"]:has(ol) > details > summary { padding: 1rem 1.3rem; }
[data-testid="stExpander"]:has(ol) > details > summary p { font-weight: 800; }
[data-testid="stExpander"]:has(ol) ol { margin: 0.3rem 0 0.7rem; padding-left: 1.3rem; }
[data-testid="stExpander"]:has(ol) li { margin: 0.6rem 0; line-height: 1.65; color: #1f2937; }
[data-testid="stExpander"]:has(ol) li::marker { color: #d97706; font-weight: 800; }

/* 3) Research Disclaimer -- softer, cleaner, compact amber emphasis */
[data-testid="stElementContainer"]:has(hr) + [data-testid="stElementContainer"]:has([data-testid="stAlert"]):has(+ [data-testid="stElementContainer"] hr) [data-testid="stAlert"] {
  background: #fffaf3 !important;
  border: 1px solid #fde8c8 !important;
  border-left: 3px solid #fbbf24 !important;
  border-radius: 10px !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  padding: 0.75rem 1.05rem !important;
  align-items: flex-start;
}

/* 4) Limitations & Interpretation -- structured research section */
[data-testid="stElementContainer"]:has(h4):has(+ [data-testid="stElementContainer"] ol) {
  border-left: 3px solid #f59e0b;
  padding-left: 0.9rem;
}
[data-testid="stElementContainer"]:has(h4):has(+ [data-testid="stElementContainer"] ol) h4 {
  font-size: 0.95rem;
  font-weight: 800;
  letter-spacing: 0.03em;
  color: #0f172a;
  margin: 0.35rem 0 0.45rem;
}
[data-testid="stElementContainer"]:has(h4) + [data-testid="stElementContainer"]:has(ol) {
  background: #f8fafc;
  border: 1px solid #e5e9f0;
  border-radius: 12px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  padding: 0.85rem 1.15rem 0.7rem;
  margin-bottom: 0.35rem;
}
[data-testid="stElementContainer"]:has(h4) + [data-testid="stElementContainer"]:has(ol) ol { margin: 0.15rem 0 0.35rem; padding-left: 1.25rem; }
[data-testid="stElementContainer"]:has(h4) + [data-testid="stElementContainer"]:has(ol) li { margin: 0.5rem 0; line-height: 1.6; color: #1f2937; }
[data-testid="stElementContainer"]:has(h4) + [data-testid="stElementContainer"]:has(ol) li::marker { color: #d97706; font-weight: 700; }
[data-testid="stElementContainer"]:has(h4) + [data-testid="stElementContainer"]:has(ol) + [data-testid="stElementContainer"] { padding: 0.1rem 0.2rem 0.75rem; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html = True)

def H(html: str):
    st.markdown(html, unsafe_allow_html = True)


def hr():
    H('<hr class="sec-divider">')


def section(label: str, heading: str, subtitle: str = ''):
    H(f'''<div class="sec-label">{label}</div><div class="sec-heading">{heading}</div>''' + (f'''<div class="sec-sub">{subtitle}</div>''' if subtitle else ''))


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

MODEL_VARIANT = 'cifake'
_VARIANT_ARTIFACTS = {
    'cifake': {
        'head': 'models/dinov2_authenticity_head.pth',
        'calibration': 'models/calibration_params.json',
        'ood': 'models/ood_params.json',
        'label': 'CIFAKE (Original)',
        'dataset_name': 'CIFAKE',
        'badge': 'CIFAKE',
        'preset_root': os.path.join('dataset', 'test'),
        'benchmark_caption': 'CIFAKE benchmark',
        'native_note': '32×32 px native resolution, not a high-resolution image',
        'scope_lines': [
            ('Dataset', 'CIFAKE'),
            ('Classes', 'Authentic / AI-Edited'),
            ('Benchmark source', '<code>dataset/test</code> — curated presets')],
        'limit_scope': 'CIFAKE (Stable Diffusion v1.4 vs CIFAR-10 natural imagery). Generalisation to commercial generators (Midjourney v6, DALL-E 3, Flux) is not guaranteed.',
        'limit_deg': 'Extreme blur (r=3.0 → −26.6% accuracy) and lossy downscaling (0.25× → −26.0%) significantly impair detection.',
        'limit_tta': '~2.01× CPU latency for +0.40% accuracy gain.' },
    'cf': {
        'head': 'models/dinov2_authenticity_head_cf.pth',
        'calibration': 'models/calibration_params_cf.json',
        'ood': 'models/ood_params_cf.json',
        'label': 'Community Forensics-Small (Research)',
        'dataset_name': 'Community Forensics-Small',
        'badge': 'CF',
        'preset_root': os.path.join('dataset', 'cf_small', 'test'),
        'benchmark_caption': 'Community Forensics-Small held-out test',
        'native_note': 'held-out test image, native source resolution',
        'scope_lines': [
            ('Dataset', 'Community Forensics-Small'),
            ('Train', '1,400'),
            ('Validation', '200'),
            ('Held-out Test', '400'),
            ('Classes', 'Authentic / AI-Edited'),
            ('Test composition', '200 / 200')],
        'limit_scope': 'Community Forensics-Small (700/700 train, 100/100 validation, held-out 200/200 test). Evaluation is reported on the untouched 400-image test split.',
        'limit_deg': 'Robustness stress tests (blur / downscaling) were run on the earlier CIFAKE pipeline; CF-specific stress tests were not run.',
        'limit_tta': '~2.01× CPU latency (measured on the earlier CIFAKE evaluation).' } }

def _artifact_paths(variant: str) -> dict:
    '''Return artifact paths for a model variant (falls back to cifake).'''
    return _VARIANT_ARTIFACTS.get(variant, _VARIANT_ARTIFACTS['cifake'])

@st.cache_resource
def load_verification_pipeline(variant: str = 'cifake'):
    """Caches pipeline resources for fast subsequent inference.

    The DINOv2 backbone is shared across variants; only the classification
    head, calibration and OOD parameters differ.
    """
    device = torch.device('cpu')
    paths = _artifact_paths(variant)
    attributor = DINOv2PatchAttributor(checkpoint_path = paths['head'], device = device)
    calib_path = paths['calibration']
    temp = 1.0
    if os.path.exists(calib_path):
        with open(calib_path, 'r', encoding = 'utf-8') as f:
            cdata = json.load(f)
            temp = float(cdata.get('temperature', 1.0))
    ood_path = paths['ood']
    ood_params = None
    if os.path.exists(ood_path):
        with open(ood_path, 'r', encoding = 'utf-8') as f:
            ood_params = json.load(f)
    return (attributor, temp, ood_params, device, paths)

def compute_ood_score(features_np: np.ndarray, ood_params: dict):
    '''Computes minimum cosine distance to training class centroids.'''
    if ood_params is None:
        return (0.0, 0.6673, False)
    mu_auth = np.array(ood_params['centroid_authentic'], dtype = np.float32)
    mu_synth = np.array(ood_params['centroid_synthetic'], dtype = np.float32)
    feat_norm = np.linalg.norm(features_np) + 1e-12
    cos_auth = np.dot(features_np, mu_auth) / (feat_norm * (np.linalg.norm(mu_auth) + 1e-12))
    cos_synth = np.dot(features_np, mu_synth) / (feat_norm * (np.linalg.norm(mu_synth) + 1e-12))
    dist_auth = 1.0 - float(cos_auth)
    dist_synth = 1.0 - float(cos_synth)
    min_dist = min(dist_auth, dist_synth)
    threshold = float(ood_params.get('threshold_95', 0.6673))
    is_ood = min_dist > threshold
    return (min_dist, threshold, is_ood)

def main():
    st.sidebar.markdown('---')
    st.sidebar.markdown('<div style="font-size:0.62rem;font-weight:800;letter-spacing:0.14em;text-transform:uppercase;color:#7dd3fc;padding:0.2rem 0 0.3rem">🧠 &nbsp;Model Variant</div>', unsafe_allow_html = True)
    variant_labels = {v: _VARIANT_ARTIFACTS[v]['label'] for v in _VARIANT_ARTIFACTS}
    selected_variant = st.sidebar.radio('Active Model', list(_VARIANT_ARTIFACTS.keys()), format_func = (lambda v: variant_labels[v]), label_visibility = 'collapsed', help = 'Select which trained model to use for inference.')
    st.sidebar.caption(f'''Active: **{variant_labels[selected_variant]}**''')
    (attributor, temperature, ood_params, device, _cached_paths) = load_verification_pipeline(selected_variant)
    paths = _artifact_paths(selected_variant)
    _badge_bg = '#16a34a' if selected_variant == 'cifake' else '#0284c7'
    H(f'''\n    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#1e40af 100%);\n                border-radius:14px;padding:1.6rem 2rem 1.4rem;margin-bottom:1.2rem;\n                border:1px solid rgba(147,197,253,0.25);\n                box-shadow:0 6px 28px rgba(15,23,42,0.28)">\n      <div style="display:flex;justify-content:space-between;align-items:flex-start;\n                  gap:1rem;flex-wrap:wrap">\n        <div style="font-size:0.6rem;font-weight:800;letter-spacing:0.18em;\n                    text-transform:uppercase;color:#93c5fd;margin-bottom:0.35rem">\n          Explainable AI &mdash; Final Year Capstone\n        </div>\n        <div style="background:{_badge_bg};color:#fff;font-size:0.6rem;font-weight:800;\n                    letter-spacing:0.1em;text-transform:uppercase;\n                    padding:0.3rem 0.75rem;border-radius:999px;white-space:nowrap;\n                    box-shadow:0 3px 10px rgba(0,0,0,0.3)">\n          Active Model &middot; {paths['badge']} &nbsp;&mdash;&nbsp; {paths['dataset_name']}\n        </div>\n      </div>\n      <div style="font-size:1.55rem;font-weight:800;color:#f8fafc;line-height:1.2;margin-bottom:0.3rem">\n        🛡️ Image Authenticity Verification &amp; AI Edit Detection\n      </div>\n      <div style="font-size:0.78rem;color:#94a3b8;line-height:1.5">\n        Vision Foundation Model &nbsp;&bull;&nbsp; DINOv2-Small (ViT-S/14) &nbsp;&bull;&nbsp;\n        Calibrated MLP Head &nbsp;&bull;&nbsp; Gradient Attribution &nbsp;&bull;&nbsp;\n        OOD Diagnostics &nbsp;&bull;&nbsp; Frequency Analysis\n      </div>\n    </div>\n    ''')
    st.sidebar.markdown('<div style="font-size:0.62rem;font-weight:800;letter-spacing:0.14em;text-transform:uppercase;color:#7dd3fc;padding:0.5rem 0 0.3rem">📁 &nbsp;Image Source</div>', unsafe_allow_html = True)
    source_choice = st.sidebar.radio('Input Source', [
        'Upload Custom Image',
        'Dataset Image Browser',
        'Benchmark Test Preset'], label_visibility = 'collapsed')
    img_input = None
    benchmark_true_class = None
    ds_meta = None
    if source_choice == 'Upload Custom Image':
        uploaded_file = st.sidebar.file_uploader('Upload an Image', type = [
            'jpg',
            'jpeg',
            'png'])
        if uploaded_file is not None:
            img_input = Image.open(uploaded_file).convert('RGB')
    elif source_choice == 'Dataset Image Browser':
        ds_base = os.path.join('dataset', 'cf_small') if selected_variant == 'cf' else 'dataset'
        _split_map = {
            'Train': 'train',
            'Validation': 'val',
            'Held-out Test': 'test' }
        st.sidebar.markdown(f'''<div style="background:rgba(30,41,59,0.75);border:1px solid #334155;border-radius:10px;padding:0.5rem 0.7rem;margin-bottom:0.35rem;font-size:0.7rem;color:#e2e8f0">🗄️ <b>{paths['dataset_name']}</b> &mdash; dataset browser<br><span style="color:#94a3b8;font-size:0.64rem">Demo/inspect individual dataset images (not benchmark metrics)</span></div>''', unsafe_allow_html = True)
        split_label = st.sidebar.radio('Split', list(_split_map), key = 'ds_split')
        cls_opts = {
            'All': None,
            'Authentic': 'authentic',
            'AI-Edited': 'ai_edited' }
        cls_label = st.sidebar.selectbox('Class', list(cls_opts), key = 'ds_cls')
        ds_pool = []
        for _c in ([cls_opts[cls_label]] if cls_opts[cls_label] else ['authentic', 'ai_edited']):
            _d = os.path.join(ds_base, _split_map[split_label], _c)
            if os.path.isdir(_d):
                ds_pool += [(_c, os.path.join(_d, _f)) for _f in sorted(os.listdir(_d)) if os.path.isfile(os.path.join(_d, _f))]
        if not ds_pool:
            st.sidebar.info(f'''No images found at `{os.path.join(ds_base, _split_map[split_label])}` for this variant.''')
        else:
            ds_idx = st.sidebar.slider('Image index', 0, len(ds_pool) - 1, 0, help = f'''Browse all {len(ds_pool)} images in this split/class filter.''')
            (_sel_cls, _sel_path) = ds_pool[ds_idx]
            if os.path.exists(_sel_path):
                _prev = Image.open(_sel_path).convert('RGB')
                st.sidebar.image(_prev, use_container_width = True)
                st.sidebar.markdown(f'''<div style="background:rgba(30,41,59,0.75);border:1px solid #334155;border-radius:10px;padding:0.5rem 0.7rem;margin:0.3rem 0 0.4rem;font-size:0.68rem;color:#e2e8f0;line-height:1.7"><b style="color:#7dd3fc">SELECTED IMAGE</b><br>File: <b>{os.path.basename(_sel_path)}</b><br>Split: <b>{split_label}</b><br>True class: <b>{_sel_cls}</b><br>Resolution: <b>{_prev.width}&times;{_prev.height} px</b></div>''', unsafe_allow_html = True)
                if st.sidebar.button('🖼 Use This Dataset Image', type = 'primary', use_container_width = True):
                    st.session_state['ds_active'] = {
                        'variant': selected_variant,
                        'path': _sel_path,
                        'meta': (split_label, _sel_cls, os.path.basename(_sel_path)) }
                _act = st.session_state.get('ds_active')
                if _act and _act['variant'] == selected_variant and os.path.exists(_act['path']):
                    img_input = Image.open(_act['path']).convert('RGB')
                    ds_meta = _act['meta']
                    st.sidebar.caption(f'''✅ **Active dataset image:** `{ds_meta[2]}` ({ds_meta[0]} · {ds_meta[1]})''')
                else:
                    st.sidebar.caption('👆 Preview above — click **Use This Dataset Image** to run verification on it.')
    elif selected_variant == 'cf':
        cf_test_root = paths['preset_root']
        cf_pool = []
        for _cls in ('authentic', 'ai_edited'):
            _dir = os.path.join(cf_test_root, _cls)
            if os.path.isdir(_dir):
                cf_pool += [(_cls, os.path.join(_dir, _f)) for _f in sorted(os.listdir(_dir)) if os.path.isfile(os.path.join(_dir, _f))]
        if not cf_pool:
            st.sidebar.error(f'''CF test split not found at `{cf_test_root}`.''')
        else:
            st.sidebar.markdown(f'''<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:0.6rem 0.8rem;margin-bottom:0.4rem"><div style="font-size:0.6rem;font-weight:800;letter-spacing:0.1em;text-transform:uppercase;color:#1d4ed8">🧪 Held-out Test Set &mdash; {len(cf_pool)} images</div><div style="font-size:0.68rem;color:#1e40af;margin-top:0.15rem">Untouched split &bull; not used for training, calibration or OOD fitting</div></div>''', unsafe_allow_html = True)
            _cls_counts = {'All': len(cf_pool), 'Authentic': sum(1 for c, _ in cf_pool if c == 'authentic'), 'AI-Edited': sum(1 for c, _ in cf_pool if c == 'ai_edited')}
            _cls_map = {'All': None, 'Authentic': 'authentic', 'AI-Edited': 'ai_edited'}
            _filt_label = st.sidebar.selectbox('Class', list(_cls_counts), format_func = (lambda k: f'{k} ({_cls_counts[k]})'))
            _filt = _cls_map[_filt_label]
            _view = [x for x in cf_pool if _filt is None or x[0] == _filt]
            _idx = st.sidebar.slider('Image index', 0, len(_view) - 1, 0, help = f'''Browse all {len(_view)} images in the current class filter.''')
            (benchmark_true_class, preset_path) = _view[_idx]
            if os.path.exists(preset_path):
                img_input = Image.open(preset_path).convert('RGB')
                st.sidebar.caption(f'''📂 `{preset_path}`''')
                st.sidebar.caption(f'''True class: **{benchmark_true_class}** · **{img_input.width}×{img_input.height} px**''')
            else:
                st.sidebar.error('Test image not found.')
    else:
        preset_options = {
            'Authentic — Test Set': os.path.join('dataset', 'test', 'authentic', '0005 (8).jpg'),
            'AI-Generated — Test Set': os.path.join('dataset', 'test', 'ai_edited', '1011 (4).jpg'),
            'Borderline — Authentic (misclassified)': os.path.join('dataset', 'test', 'authentic', '0064 (3).jpg'),
            'Borderline — Synthetic (misclassified)': os.path.join('dataset', 'test', 'ai_edited', '1040.jpg') }
        selected_preset = st.sidebar.selectbox('Select Test Image', list(preset_options.keys()))
        preset_path = preset_options[selected_preset]
        if os.path.exists(preset_path):
            img_input = Image.open(preset_path).convert('RGB')
            benchmark_true_class = os.path.basename(os.path.dirname(preset_path))
            st.sidebar.caption(f'📂 `{preset_path}`')
        else:
            st.sidebar.error('Preset file not found.')
    st.sidebar.markdown('---')
    st.sidebar.markdown('<div style="font-size:0.62rem;font-weight:800;letter-spacing:0.14em;text-transform:uppercase;color:#7dd3fc;padding:0.2rem 0 0.3rem">⚙️ &nbsp;Analysis Options</div>', unsafe_allow_html = True)
    enable_tta = st.sidebar.checkbox('Enable Test-Time Augmentation (TTA)', value = False, help = 'Averages predictions of original and horizontally flipped views. ~2× CPU latency.')
    show_frequency = st.sidebar.checkbox('Show Frequency-Domain Diagnostic (FFT)', value = True, help = 'Supplementary 2D FFT spectrum. Does NOT influence the verdict.')
    st.sidebar.markdown('---')
    st.sidebar.markdown('<div style="font-size:0.62rem;font-weight:800;letter-spacing:0.14em;text-transform:uppercase;color:#7dd3fc;padding:0.2rem 0 0.2rem">ℹ️ &nbsp;System</div>', unsafe_allow_html = True)
    st.sidebar.caption('Backbone: `facebook/dinov2-small` (DINOv2-Small / ViT-S/14)')
    st.sidebar.caption(f'''Active model: **{paths['label']}**''')
    st.sidebar.caption(f'''Dataset: **{paths['dataset_name']}**''')
    st.sidebar.caption(f'''Head: `{paths['head']}`''')
    st.sidebar.caption(f'''Calibration T = `{temperature:.4f}`''')
    if ood_params is not None:
        st.sidebar.caption(f'''OOD 95th-pct threshold = `{float(ood_params.get('threshold_95', 0.0)):.4f}`''')
    if paths.get('ood') and os.path.exists(paths['ood']):
        st.sidebar.caption(f'''OOD params: `{paths['ood']}`''')
    st.sidebar.caption('Features: 384-D CLS · Patch grid: 16×16')
    st.sidebar.caption('Hardware: CPU inference')
    st.sidebar.markdown('---')
    st.sidebar.markdown('<div style="font-size:0.62rem;font-weight:800;letter-spacing:0.14em;text-transform:uppercase;color:#7dd3fc;padding:0.2rem 0 0.3rem">📊 &nbsp;Dataset / Evaluation Scope</div>', unsafe_allow_html = True)
    _scope_rows = ''.join(f'<div style="display:flex;justify-content:space-between;gap:0.6rem;font-size:0.7rem;padding:0.12rem 0;border-bottom:1px solid rgba(148,163,184,0.15)"><span style="color:#94a3b8">{k}</span><span style="color:#f1f5f9;font-weight:600;text-align:right">{v}</span></div>' for k, v in paths['scope_lines'])
    st.sidebar.markdown(f'''<div style="background:rgba(30,41,59,0.75);border:1px solid #334155;border-radius:10px;padding:0.65rem 0.8rem;margin-bottom:0.4rem">{_scope_rows}</div>''', unsafe_allow_html = True)
    if img_input is None:
        H('\n        <div style="background:#fff;border:1.5px dashed #cbd5e1;border-radius:12px;\n                    padding:3rem 2rem;text-align:center;color:#64748b;margin-top:1.5rem">\n          <div style="font-size:2.5rem;margin-bottom:0.6rem">📂</div>\n          <div style="font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:0.3rem">\n            No Image Selected\n          </div>\n          <div style="font-size:0.78rem;color:#64748b;line-height:1.6">\n            Upload a custom image, browse the dataset, or select a benchmark preset<br>\n            from the sidebar to begin DINOv2 authenticity analysis.\n          </div>\n        </div>\n        ')
        return None
    hr()
    orig_w, orig_h = img_input.width, img_input.height
    is_benchmark = source_choice == 'Benchmark Test Preset'
    bench_caption = paths['benchmark_caption']
    if source_choice == 'Dataset Image Browser' and ds_meta:
        src_note = f'''(dataset browser — {paths['dataset_name']} · {ds_meta[0]} split · true class {ds_meta[1]} · {orig_w}×{orig_h} px native).'''
    elif not is_benchmark:
        src_note = '(custom upload).'
    elif selected_variant == 'cf':
        src_note = f'''(Community Forensics-Small held-out test set — {orig_w}×{orig_h} px native resolution).'''
    else:
        src_note = '(CIFAKE benchmark — 32×32 px native resolution, not a high-resolution image).'
    section('01 — Input', 'Input Image', f'''Source image: {orig_w}×{orig_h} px ''' + src_note)
    disp_img = _display_img(img_input, max_px = 512)
    disp_224 = img_input.resize((224, 224), Image.LANCZOS)
    (in1, in2, in3, in4) = st.columns([
        1,
        0.15,
        1,
        1.8], gap = 'small')
    with in1:
        st.image(disp_img, use_container_width = True)
        _src_badge = bench_caption if is_benchmark else f'''{paths['dataset_name']} dataset &mdash; {ds_meta[0]} split''' if ds_meta else None
        _true_cls = benchmark_true_class if is_benchmark else ds_meta[1] if ds_meta else None
        H(f'''<p class="ev-cap" style="text-align:center"><b>Original Source</b><br>{orig_w}&times;{orig_h} px''' + (f''' &nbsp;&mdash;&nbsp; <span style="color:#f59e0b;font-weight:700">{_src_badge}</span>''' if _src_badge else '') + (f'''<br><span style="color:#0f172a;font-weight:700">True class: {_true_cls}</span>''' if _true_cls else '') + '</p>')
        if is_benchmark:
            if selected_variant == 'cf':
                H(f'''<span class="res-note">🧪 <b>HELD-OUT TEST SPLIT</b> &mdash; {bench_caption}. Native resolution {orig_w}&times;{orig_h} px. This split was never used for training, calibration or OOD fitting.</span>''')
            else:
                H(f'''<span class="res-note">Native source resolution: {orig_w}&times;{orig_h} px. Display enlargement cannot recover missing source detail.</span>''')
    with in2:
        H('<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:1.4rem;color:#94a3b8">&rarr;</div>')
    with in3:
        st.image(disp_224, use_container_width = True)
        H('<p class="ev-cap" style="text-align:center"><b>Model Input Preview</b><br>224&times;224 px &mdash; DINOv2 preprocessing</p>')
        H('<span class="res-note">📌 Display preview only. Inference uses <code>attributor.transform()</code> internally &mdash; this image does not pass through the model.</span>')
    with in4:
        H(f'''\n        <div class="info-card">\n          <div class="info-card-label">Image Metadata</div>\n          <div style="font-size:0.78rem;color:#0f172a;line-height:2.1">\n            <b>Source Resolution:</b> {orig_w}&times;{orig_h} px<br>\n            <b>Colour Mode:</b> RGB<br>\n            <b>Backbone Input:</b> 224&times;224 px<br>\n            <b>Backbone:</b> <code>facebook/dinov2-small</code><br>\n            <b>Architecture:</b> ViT-S/14 (frozen, 21M params)<br>\n            <b>Calibration T:</b> {temperature:.4f}<br>\n            <b>Patch Grid:</b> 16&times;16 (256 tokens)<br>\n            <b>Feature Dim:</b> 384-D CLS token\n          </div>\n        </div>\n        ''')
    with st.spinner('Running DINOv2 inference and gradient attribution...'):
        t0 = time.perf_counter()
        explanation = attributor.explain(img_input)
        total_time = (time.perf_counter() - t0) * 1000
        input_tensor = attributor.transform(img_input).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_feat = attributor.backbone(input_tensor)
            raw_logits = attributor.head(cls_feat)
            cal_logits = raw_logits / temperature
            cal_probs = F.softmax(cal_logits, dim = -1).squeeze(0).cpu().numpy()
        classes = attributor.classes
        idx_auth = classes.index('authentic')
        idx_synth = classes.index('ai_edited')
        prob_auth = float(cal_probs[idx_auth])
        prob_synth = float(cal_probs[idx_synth])
        if prob_synth >= 0.5:
            pred_label = 'AI-Generated / Synthetic'
            pred_conf = prob_synth
            is_synthetic = True
        else:
            pred_label = 'Authentic'
            pred_conf = prob_auth
            is_synthetic = False
        entropy_val = float(compute_entropy(np.array([
            [
                prob_synth,
                prob_auth]]))[0])
        cls_feat_np = cls_feat.squeeze(0).cpu().numpy()
        (ood_dist, ood_thresh, is_ood) = compute_ood_score(cls_feat_np, ood_params)
        tta_executed = False
        tta_prob_auth = None
        tta_prob_synth = None
        if enable_tta:
            img_flipped = img_input.transpose(Image.FLIP_LEFT_RIGHT)
            flip_tensor = attributor.transform(img_flipped).unsqueeze(0).to(device)
            with torch.no_grad():
                cls_flip = attributor.backbone(flip_tensor)
                flip_logits = attributor.head(cls_flip)
                cal_flip_logits = flip_logits / temperature
                flip_probs = F.softmax(cal_flip_logits, dim = -1).squeeze(0).cpu().numpy()
            tta_prob_auth = 0.5 * (prob_auth + float(flip_probs[idx_auth]))
            tta_prob_synth = 0.5 * (prob_synth + float(flip_probs[idx_synth]))
            tta_executed = True
    hr()
    section('02 — Classification', 'Authenticity Verdict', 'Primary DINOv2 classification result. Supplementary diagnostics are in sections 03–05.')
    verdict_bg = 'linear-gradient(135deg,#dc2626,#991b1b)' if is_synthetic else 'linear-gradient(135deg,#16a34a,#15803d)'
    verdict_icon = '🔴' if is_synthetic else '🟢'
    _true_cmp = benchmark_true_class if is_benchmark else ds_meta[1] if ds_meta else None
    _cmp_html = ''
    if _true_cmp:
        _pred_cls = 'ai_edited' if is_synthetic else 'authentic'
        _ok = _true_cmp == _pred_cls
        _cmp_html = f'''<div style="font-size:0.73rem;background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3);border-radius:8px;padding:0.4rem 0.75rem;margin-bottom:0.45rem;line-height:1.6">🎯 <b>True class:</b> <span style="text-transform:uppercase">{_true_cmp}</span> &nbsp;&nbsp;|&nbsp;&nbsp; <b>Model prediction:</b> {pred_label} &nbsp;&nbsp;|&nbsp;&nbsp; ''' + ('<b style="color:#bbf7d0">✅ Correct</b>' if _ok else '<b style="color:#fecaca">❌ Incorrect</b>') + '</div>'
    if is_benchmark:
        _prov = f'''Dataset: <b>{paths['dataset_name']}</b> &nbsp;&bull;&nbsp; Split: <b>Held-out Test</b> &nbsp;&bull;&nbsp; ''' + (f'''True class: <b>{benchmark_true_class}</b> &nbsp;&bull;&nbsp; ''' if benchmark_true_class else '') + f'''OOD: <b>{'⚠️ flagged' if is_ood else '✅ in-distribution'}</b>'''
    elif ds_meta:
        _prov = f'''Dataset: <b>{paths['dataset_name']}</b> &nbsp;&bull;&nbsp; Source: <b>Dataset Image Browser</b> ({ds_meta[0]} split) &nbsp;&bull;&nbsp; True class: <b>{ds_meta[1]}</b> &nbsp;&bull;&nbsp; OOD: <b>{'⚠️ flagged' if is_ood else '✅ in-distribution'}</b>'''
    else:
        _prov = f'''Dataset: <b>{paths['dataset_name']}</b> &nbsp;&bull;&nbsp; Source: <b>custom upload</b> &nbsp;&bull;&nbsp; OOD: <b>{'⚠️ flagged' if is_ood else '✅ in-distribution'}</b>'''
    H(f'''\n    <div style="background:{verdict_bg};border-radius:14px;padding:1.15rem 1.5rem;\n                color:#fff;margin-bottom:0.9rem;border:1px solid rgba(255,255,255,0.18);\n                box-shadow:0 6px 22px rgba(0,0,0,0.22)">\n      <div style="display:flex;justify-content:space-between;align-items:center;gap:0.8rem;\n                  flex-wrap:wrap;margin-bottom:0.25rem">\n        <div style="font-size:0.58rem;font-weight:800;letter-spacing:0.14em;\n                    text-transform:uppercase;opacity:0.78">\n          Classification Verdict\n        </div>\n        <div style="font-size:0.56rem;font-weight:800;letter-spacing:0.09em;\n                    text-transform:uppercase;background:rgba(255,255,255,0.18);\n                    border:1px solid rgba(255,255,255,0.3);border-radius:999px;\n                    padding:0.18rem 0.65rem;white-space:nowrap">\n          {paths['badge']} &middot; {paths['dataset_name']}\n        </div>\n      </div>\n      <div style="font-size:1.4rem;font-weight:800;line-height:1.2;margin-bottom:0.25rem">\n        {verdict_icon} &nbsp;{pred_label}\n      </div>\n      <div style="font-size:0.75rem;opacity:0.9;display:flex;gap:1.2rem;flex-wrap:wrap;\n                  margin-bottom:0.45rem">\n        <span>Calibrated confidence: <b>{pred_conf * 100:.2f}%</b></span>\n        <span>Entropy: <b>{entropy_val:.3f} bits</b></span>\n        <span>Latency: <b>{total_time:.1f} ms</b></span>\n      </div>\n      {_cmp_html}\n      <div style="font-size:0.66rem;opacity:0.85;background:rgba(0,0,0,0.18);\n                  border-radius:7px;padding:0.32rem 0.65rem;line-height:1.5">\n        {_prov}\n      </div>\n    </div>\n    ''')
    (m1, m2, m3, m4) = st.columns(4)
    with m1:
        st.metric('Calibrated Confidence', f'''{pred_conf * 100:.2f}%''', help = f'''Post-hoc temperature-scaled probability for the predicted class (T={temperature:.4f}).''')
    with m2:
        st.metric('Predictive Entropy', f'''{entropy_val:.3f} bits''', help = 'Shannon entropy of class probabilities. <0.3 bits = certain; >0.7 bits = ambiguous.')
    with m3:
        st.metric('P(Synthetic)', f'''{prob_synth * 100:.2f}%''', help = 'Calibrated probability that the image is AI-generated. Decision threshold: 0.50.')
    with m4:
        st.metric('Inference Latency', f'''{total_time:.1f} ms''', help = 'Full forward pass + gradient attribution on CPU.')
    auth_pct = int(prob_auth * 100)
    synth_pct = int(prob_synth * 100)
    H(f'''\n    <div style="margin:0.6rem 0 0.2rem">\n      <div class="prob-row">\n        <span class="prob-label">P(Authentic)</span>\n        <div class="prob-track">\n          <div class="prob-fill-auth" style="width:{auth_pct}%"></div>\n        </div>\n        <span class="prob-val">{prob_auth * 100:.2f}%</span>\n      </div>\n      <div class="prob-row">\n        <span class="prob-label">P(Synthetic)</span>\n        <div class="prob-track">\n          <div class="prob-fill-synth" style="width:{synth_pct}%"></div>\n        </div>\n        <span class="prob-val">{prob_synth * 100:.2f}%</span>\n      </div>\n    </div>\n    ''')
    with st.expander('📊 Confidence & Uncertainty — Interpretation', expanded = True):
        (cu_c1, cu_c2, cu_c3, cu_c4) = st.columns(4)
        with cu_c1:
            st.metric('Calibrated Confidence', f'''{pred_conf * 100:.2f}%''', help = 'Post-hoc temperature-scaled probability assigned to the predicted class.')
        with cu_c2:
            st.metric('Predictive Entropy', f'''{entropy_val:.3f} bits''', help = 'Shannon entropy of calibrated class probabilities. Low = certain, High = uncertain.')
        with cu_c3:
            st.metric('P(Authentic)', f'''{prob_auth * 100:.4f}%''')
        with cu_c4:
            st.metric('P(Synthetic)', f'''{prob_synth * 100:.4f}%''')
        conf_pct = pred_conf * 100
        if conf_pct >= 90 and entropy_val < 0.3:
            st.success(f'''**High confidence ({conf_pct:.1f}%):** The calibrated model probability is strongly concentrated on the \'{pred_label}\' class. Low predictive entropy ({entropy_val:.3f} bits) indicates a stable, well-separated decision.''')
        elif conf_pct >= 70 and entropy_val < 0.7:
            st.info(f'''**Moderate confidence ({conf_pct:.1f}%):** The model leans toward \'{pred_label}\' but predictive entropy ({entropy_val:.3f} bits) is non-trivial. Consider supplementary diagnostics (OOD, TTA, FFT) before drawing conclusions.''')
        else:
            st.warning(f'''**Low confidence ({conf_pct:.1f}%):** Probabilities are close to the 0.50 decision boundary. Predictive entropy ({entropy_val:.3f} bits) is elevated — borderline case. Do not treat the verdict as reliable without additional evidence.''')
        st.caption('ℹ️ *Confidence = calibrated probability concentrated on one class. Entropy = uncertainty. Higher entropy → less certain. Confidence alone is not forensic proof.*')
    hr()
    section('03 — Explainability', 'Visual Evidence', 'How the model spatially attributed its authenticity decision.')
    img_224_display = img_input.resize((512, 512), Image.LANCZOS)
    heatmap_display = Image.fromarray(explanation['colored_heatmap']).resize((512, 512), Image.LANCZOS)
    overlay_display = explanation['overlay_image'].resize((512, 512), Image.LANCZOS)
    (ev1, ev2, ev3) = st.columns(3, gap = 'medium')
    with ev1:
        H('<div class="ev-card"><div><span class="ev-letter">A</span></div><div class="ev-title">Source / Model Input</div><div class="ev-sub">224 &times; 224 px &mdash; DINOv2 preprocessing</div></div>')
        st.image(img_224_display, use_container_width = True)
        H('<p class="ev-cap">Standardized input tensor as received by the DINOv2 backbone. Resizing changes scale — it does <em>not</em> add new image detail. Displayed with LANCZOS interpolation for visual smoothness only; the model receives its own internal transform.</p>')
    with ev2:
        H('<div class="ev-card"><div><span class="ev-letter">B</span></div><div class="ev-title">Patch Attribution Heatmap</div><div class="ev-sub">16&times;16 gradient grid &rarr; 224&times;224 bicubic</div></div>')
        st.image(heatmap_display, use_container_width = True)
        H('<div class="attr-legend"><span style="font-size:0.65rem;color:#64748b;white-space:nowrap">Low&nbsp;relevance</span><div class="al-bar"></div><span style="font-size:0.65rem;color:#64748b;white-space:nowrap">High&nbsp;relevance</span></div><p class="ev-cap" style="text-align:center">Gradient-based patch relevance scores upsampled from the 16&times;16 ViT patch grid. Indicates <em>model attention</em>, not ground-truth tampering regions.</p>')
    with ev3:
        H('<div class="ev-card"><div><span class="ev-letter">C</span></div><div class="ev-title">Attribution Overlay</div><div class="ev-sub">Heatmap composited over model input</div></div>')
        st.image(overlay_display, use_container_width = True)
        H('<p class="ev-cap">Heatmap overlaid on the model input image. Brighter/warmer regions received higher gradient attribution. <em>Highlighted areas do not constitute forensic evidence of manipulation.</em></p>')
    H(f'''<p style="font-size:0.68rem;color:#94a3b8;margin-top:0.3rem">⚠️ Attribution is approximate gradient-based analysis relative to the model\'s training distribution ({paths['dataset_name']}). It is not a pixel-level ground-truth tampering mask.</p>''')
    hr()
    section('04 — Diagnostics', 'Model Diagnostics', 'Supplementary indicators. These do NOT override the primary classifier verdict.')
    (ood_col1, ood_col2) = st.columns([
        1,
        2], gap = 'medium')
    with ood_col1:
        gauge_max = max(ood_thresh * 1.5, ood_dist * 1.1, 1.0)
        dist_pct = min((ood_dist / gauge_max) * 100, 100)
        thresh_pct = min((ood_thresh / gauge_max) * 100, 100)
        ood_color = '#ef4444' if is_ood else '#16a34a'
        H(f'''\n        <div class="info-card">\n          <div class="info-card-label">Out-of-Distribution (OOD) Indicator</div>\n          <div style="font-size:1.5rem;font-weight:800;color:{ood_color};margin-bottom:0.05rem">\n            {ood_dist:.4f}\n          </div>\n          <div style="font-size:0.7rem;color:#64748b;margin-bottom:0.1rem">\n            Cosine centroid distance\n          </div>\n          <div class="ood-track">\n            <div class="ood-thresh-line" style="left:{thresh_pct:.1f}%"></div>\n            <div class="ood-needle"      style="left:{dist_pct:.1f}%"></div>\n          </div>\n          <div style="display:flex;justify-content:space-between;font-size:0.6rem;color:#94a3b8">\n            <span>0</span>\n            <span>Threshold {ood_thresh:.4f}</span>\n            <span>{gauge_max:.2f}</span>\n          </div>\n          <div style="font-size:0.67rem;color:#64748b;margin-top:0.4rem">\n            95th-percentile threshold: <b>{ood_thresh:.4f}</b><br>\n            Status: <b style="color:{ood_color}">\n              {'⚠️ Atypical (OOD)' if is_ood else '✅ In-Distribution'}\n            </b>\n          </div>\n        </div>\n        ''')
    with ood_col2:
        st.markdown('**What is the OOD Indicator?**')
        st.markdown('The 384-D DINOv2 feature vector is compared to the training-set class centroids using cosine distance. A distance exceeding the 95th-percentile validation threshold flags the representation as atypical.')
        if is_ood:
            st.warning(f'''⚠️ **Atypical Representation (OOD Flag):** Distance `{ood_dist:.4f}` exceeds the 95th-percentile threshold `{ood_thresh:.4f}`.\n\n*An OOD flag means the image\'s feature representation is unusual relative to the training distribution. It does NOT automatically prove the image is synthetic or tampered.*''')
        else:
            st.success(f'''✅ **In-Distribution:** Distance `{ood_dist:.4f}` is within the 95th-percentile threshold `{ood_thresh:.4f}`. The image feature representation is consistent with the training distribution.''')
    if tta_executed:
        hr()
        section('04b — Robustness', 'Robustness Check — Test-Time Augmentation', 'Two-view aggregation: 0.50 × Original + 0.50 × Horizontal Flip. Robustness diagnostic only — does NOT change the primary verdict.')
        orig_verdict = 'AI-Generated / Synthetic' if prob_synth >= 0.5 else 'Authentic'
        tta_verdict = 'AI-Generated / Synthetic' if tta_prob_synth >= 0.5 else 'Authentic'
        agree = orig_verdict == tta_verdict
        delta = (tta_prob_synth - prob_synth) * 100
        (tta_c1, tta_c2, tta_c3, tta_c4) = st.columns(4, gap = 'small')
        with tta_c1:
            H(f'''\n            <div class="info-card">\n              <div class="info-card-label">Single-Crop (Original)</div>\n              <div style="font-size:0.8rem;font-weight:700;color:#0f172a;margin-bottom:0.4rem">\n                {orig_verdict}\n              </div>\n              <div style="font-size:0.72rem;color:#64748b;line-height:1.8">\n                P(Auth): <b>{prob_auth * 100:.2f}%</b><br>\n                P(Synth): <b>{prob_synth * 100:.2f}%</b>\n              </div>\n            </div>\n            ''')
        with tta_c2:
            H(f'''\n            <div class="info-card">\n              <div class="info-card-label">TTA — Averaged Views</div>\n              <div style="font-size:0.8rem;font-weight:700;color:#0f172a;margin-bottom:0.4rem">\n                {tta_verdict}\n              </div>\n              <div style="font-size:0.72rem;color:#64748b;line-height:1.8">\n                P(Auth): <b>{tta_prob_auth * 100:.2f}%</b><br>\n                P(Synth): <b>{tta_prob_synth * 100:.2f}%</b>\n              </div>\n            </div>\n            ''')
        with tta_c3:
            agree_color = '#16a34a' if agree else '#dc2626'
            agree_text = f'''✅ Agreement<br>{tta_verdict}''' if agree else '⚠️ Disagreement<br>Views differ'
            H(f'''\n            <div class="info-card" style="border-color:{agree_color}40">\n              <div class="info-card-label">Prediction Agreement</div>\n              <div style="font-size:0.82rem;font-weight:700;color:{agree_color};line-height:1.4">\n                {agree_text}\n              </div>\n            </div>\n            ''')
        with tta_c4:
            shift_color = '#dc2626' if delta > 3 else '#eab308' if abs(delta) > 1 else '#16a34a'
            H(f'''\n            <div class="info-card">\n              <div class="info-card-label">Probability Shift (Synthetic)</div>\n              <div style="font-size:1.3rem;font-weight:800;color:{shift_color}">\n                {delta:+.2f}%\n              </div>\n              <div style="font-size:0.67rem;color:#94a3b8;margin-top:0.2rem">\n                TTA avg minus single-crop\n              </div>\n            </div>\n            ''')
        if agree and abs(delta) < 3.0:
            st.info(f'''ℹ️ Both views agree (`{orig_verdict}`) with a small shift ({delta:+.2f}%). Prediction is relatively stable under horizontal reflection.''')
        elif agree:
            st.warning(f'''ℹ️ Both views agree (`{orig_verdict}`) but show a notable shift ({delta:+.2f}%). Some sensitivity to horizontal reflection detected.''')
        else:
            st.error(f'''⚠️ Views disagree (shift: {delta:+.2f}%). High decision sensitivity near the 0.50 boundary. Treat the result with additional caution.''')
    if show_frequency:
        hr()
        section('05 — Frequency', 'Frequency-Domain Diagnostic Analysis', 'Supplementary 2D FFT analysis. The DINOv2 CLS head is the primary classifier — FFT is orthogonal and does NOT influence the verdict.')
        fft_gray = img_input.convert('L').resize((224, 224))
        fft_arr = np.asarray(fft_gray, dtype = np.float64) / 255.0
        fft_2d = np.fft.fftshift(np.fft.fft2(fft_arr))
        log_mag = np.log1p(np.abs(fft_2d))
        norm_spec = (((log_mag - log_mag.min()) / ((log_mag.max() - log_mag.min()) + 1e-12)) * 255).astype(np.uint8)
        fft_feats = extract_compact_fft_features(img_input)
        radial_8 = fft_feats[:8]
        (f_col1, f_col2) = st.columns([
            1,
            1], gap = 'medium')
        with f_col1:
            fft_display = Image.fromarray(norm_spec).resize((512, 512), Image.LANCZOS)
            st.image(fft_display, caption = '2D FFT Log-Magnitude Spectrum (DC Centred, Grayscale)', use_container_width = True)
            H('<p class="ev-cap">Spatial frequency distribution. Natural images typically show a smooth 1/f roll-off; AI-generated images may exhibit unusual spectral patterns. Diagnostic only.</p>')
        with f_col2:
            (fig, ax) = plt.subplots(figsize = (7, 4.2), dpi = 200)
            fig.patch.set_facecolor('white')
            colors = [
                '#3b82f6'] * 4 + [
                '#f59e0b'] * 4
            ax.bar(range(1, 9), radial_8 * 100, color = colors, alpha = 0.88, width = 0.7)
            ax.set_xlabel('Radial Frequency Band (1 = Low / DC, 8 = Nyquist High)', fontsize = 11)
            ax.set_ylabel('Energy Proportion (%)', fontsize = 11)
            ax.set_title('Radial Spectral Energy Partition — 8 Bands', fontsize = 12, fontweight = 'bold')
            ax.set_xticks(range(1, 9))
            ax.tick_params(axis = 'both', labelsize = 10)
            ax.grid(True, alpha = 0.25, axis = 'y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            H('<p class="ev-cap">Blue bars = low-frequency bands (dominant in most images). Amber bars = high-frequency bands. Energy distribution alone is insufficient for classification.</p>')
    hr()
    with st.expander('06 — Technical Details & Specifications', expanded = False):
        st.markdown(f'''\n| Property | Value |\n|---|---|\n| **Backbone** | DINOv2-Small (`facebook/dinov2-small`) |\n| **Architecture** | Vision Transformer ViT-S/14 |\n| **Backbone Params** | ~21M (frozen) |\n| **Input Resolution** | 224 × 224 px (ImageNet normalisation) |\n| **Feature Dimension** | 384-D CLS token |\n| **Patch Grid** | 16 × 16 (256 spatial tokens) |\n| **Head** | Linear(384→256)→ReLU→Dropout(0.1)→Linear(256→2) |\n| **Calibration** | Temperature Scaling (T = {temperature:.4f}) |\n| **Attribution** | Gradient w.r.t. patch tokens, bicubic upsample |\n| **OOD Method** | Min cosine centroid distance, 95th-pct threshold |\n| **Frequency** | 2D FFT log-magnitude, 32-D radial energy features |\n| **Hardware** | CPU |\n''')
    with st.expander('⚠️ Known Model Limitations', expanded = False):
        st.markdown(f'''\n1. **Dataset Scope:** {paths['limit_scope']}\n2. **No Ground-Truth Masks:** Gradient attribution shows model attention, not human-annotated tampered pixels.\n3. **Degradation Sensitivity:** {paths['limit_deg']}\n4. **OOD Benchmark:** No external OOD benchmark used; verification relies on proxy manifold testing.\n5. **TTA Overhead:** {paths['limit_tta']}\n''')
    hr()
    st.warning(f'''⚠️ **Research Disclaimer:** This system provides a probabilistic AI-based authenticity assessment, not definitive forensic proof. Attribution visualizations represent approximate model attention relative to the {paths['dataset_name']} training distribution. OOD detection has not been validated on an external real-world benchmark.''')
    hr()
    st.markdown('#### ⚠️ Limitations & Interpretation')
    st.markdown('1. Prediction is not forensic proof.\n2. Confidence indicates model certainty, not an authenticity guarantee.\n3. Explainability heatmap shows model attribution, not confirmed manipulation location.\n4. OOD indicates unfamiliarity with the training distribution; it does not by itself mean an image is fake.\n5. Performance may vary on unseen generators, editing methods, and real-world images.\n6. Evaluation is based on the Community Forensics-Small dataset and its held-out test set; broader external validation is still required.')
    st.markdown('**Research scope:** This system is intended as an explainable research/decision-support tool, not a standalone forensic or clinical-grade certification system.')


if __name__ == "__main__":
    main()
