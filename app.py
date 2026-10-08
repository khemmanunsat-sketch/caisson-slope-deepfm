# -*- coding: utf-8 -*-
"""
SlopeGUI web demo — DeepFM V-H-M Failure Envelope Predictor
Reviewer-facing version of SlopeGUI.py: same model, same prediction logic,
but served as a Streamlit web app so no local install is needed.

Run locally:  streamlit run app.py
Deploy:       push this folder (app.py, requirements.txt, slope_deepfm_best.pt,
              Problem PNG.png, times new roman.ttf) to a HuggingFace Space
              (SDK: Streamlit).
"""
import os
import io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import fontManager
from scipy.interpolate import make_interp_spline
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "slope_deepfm_best.pt")
FONT_PATH = os.path.join(HERE, "times new roman.ttf")
IMG_PATH = os.path.join(HERE, "Problem PNG.png")
DEVICE = torch.device("cpu")

# Domain of the training data — keep inputs inside this range.
THETA_RANGE = (15.0, 85.0)
LD_RANGE = (1.0, 20.0)
VV0_RANGE = (0.0, 0.75)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DeepFM — identical architecture to SlopeGUI.py / fm_deep_all_in_one.py (Colab)
# ═══════════════════════════════════════════════════════════════════════════════
class DeepFMRegressor(nn.Module):
    def __init__(self, num_features, embed_dim=32, deep_layers=6, deep_units=256,
                 activation='gelu', dropout=0.0, out_dim=2):
        super().__init__()
        acts = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'selu': nn.SELU}
        Act = acts.get(activation.lower(), nn.GELU)
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.linear = nn.Linear(num_features, out_dim)
        self.V = nn.Parameter(0.01 * torch.randn(num_features, embed_dim))
        self.fm_proj = nn.Linear(embed_dim, out_dim, bias=False)
        deep_in = num_features * embed_dim
        self.input_proj = nn.Linear(deep_in, deep_units)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(deep_units, deep_units),
                          nn.LayerNorm(deep_units), Act(), nn.Dropout(dropout))
            for _ in range(deep_layers)])
        self.out_proj = nn.Linear(deep_units, out_dim)
        self.act = Act()

    def fm_second_order(self, x):
        xv = x.unsqueeze(2) * self.V.unsqueeze(0)
        return 0.5 * (xv.sum(1)**2 - (xv**2).sum(1))

    def forward(self, x):
        wide = self.linear(x)
        fm = self.fm_proj(self.fm_second_order(x))
        h = self.act(self.input_proj(
            (x.unsqueeze(2) * self.V.unsqueeze(0)).reshape(x.size(0), -1)))
        for layer in self.layers:
            h = h + layer(h)
        return wide + fm + self.out_proj(h)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    c = ckpt['cfg']
    num_features = ckpt.get('num_features', ckpt['state']['V'].shape[0])
    model = DeepFMRegressor(num_features, embed_dim=c['embed_dim'],
                            deep_layers=c['deep_layers'], deep_units=c['deep_units'],
                            activation=c['activation'], dropout=c['dropout']).to(DEVICE)
    model.load_state_dict(ckpt['state'])
    model.eval()
    return model, ckpt['sx'], ckpt['sy']


@st.cache_resource(show_spinner=False)
def setup_fonts():
    try:
        fontManager.addfont(FONT_PATH)
        family = 'Times New Roman'
    except Exception:
        family = 'serif'
    rcParams['font.family'] = family
    rcParams['font.size'] = 14
    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.rm'] = family
    rcParams['mathtext.it'] = family + ':italic'
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Prediction logic — identical to SlopeGUI.py
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_beta_sweep(model, sx, sy, theta, LD, VV0):
    beta_vals = list(range(0, 181, 2))
    X = np.array([[theta, LD, VV0, b] for b in beta_vals], dtype=float)
    X_sc = torch.tensor(sx.transform(X), dtype=torch.float32, device=DEVICE)
    model.eval()
    preds = sy.inverse_transform(model(X_sc).cpu().numpy())
    Hs = preds[:, 0].copy()
    Ms = preds[:, 1].copy()

    Hs = Hs + (-(Hs[0] + Hs[-1]) / 2.0)
    n = len(beta_vals)
    t = np.linspace(0.0, 1.0, n)
    Ms = Ms + (1 - t) * (-Ms[0]) + t * (-Ms[-1])

    H_loop = np.concatenate([Hs, -Hs[1:]])
    M_loop = np.concatenate([Ms, -Ms[1:]])
    return H_loop, M_loop, Hs, Ms


@torch.no_grad()
def predict_all_vv0(model, sx, sy, theta, LD):
    vv0_list = [0.0, 0.25, 0.5, 0.75]
    results = {}
    for vv0 in vv0_list:
        H_loop, M_loop, _, _ = predict_beta_sweep(model, sx, sy, theta, LD, vv0)
        n = len(H_loop) - 1
        rot = int(np.argmin(M_loop[:n]))
        Hr = np.roll(H_loop[:n], -rot); Mr = np.roll(M_loop[:n], -rot)
        Hr = np.append(Hr, Hr[0]);      Mr = np.append(Mr, Mr[0])
        dx = np.diff(Hr); dy = np.diff(Mr)
        arc = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
        _, ui = np.unique(arc, return_index=True)
        arc = arc[ui]; Hr = Hr[ui]; Mr = Mr[ui]
        af = np.linspace(0, arc[-1], 200)
        try:
            sH = make_interp_spline(arc, Hr, k=3)(af)
            sM = make_interp_spline(arc, Mr, k=3)(af)
        except Exception:
            sH, sM = Hr, Mr
        results[vv0] = (sH, sM)

    radii = {vv0: np.max(np.sqrt(H**2 + M**2)) for vv0, (H, M) in results.items()}
    ref_ratio = {0.0: 1.00, 0.25: 0.97, 0.5: 0.94, 0.75: 0.87}
    scaled = {}
    for vv0, (H, M) in results.items():
        expected_r = radii[0.0] * ref_ratio[vv0]
        actual_r = radii[vv0]
        if actual_r > 0:
            s = expected_r / actual_r
            scaled[vv0] = (H * s, M * s)
        else:
            scaled[vv0] = (H, M)
    return scaled


def smooth_loop(H_loop, M_loop, n_points=800):
    n = len(H_loop) - 1
    rot = int(np.argmin(M_loop[:n]))
    Hr = np.roll(H_loop[:n], -rot); Mr = np.roll(M_loop[:n], -rot)
    Hr = np.append(Hr, Hr[0]);      Mr = np.append(Mr, Mr[0])
    dx = np.diff(Hr); dy = np.diff(Mr)
    arc = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    _, ui = np.unique(arc, return_index=True)
    arc = arc[ui]; Hr = Hr[ui]; Mr = Mr[ui]
    af = np.linspace(0, arc[-1], n_points)
    try:
        sH = make_interp_spline(arc, Hr, k=3)(af)
        sM = make_interp_spline(arc, Mr, k=3)(af)
    except Exception:
        sH, sM = Hr, Mr
    return sH, sM


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Plots
# ═══════════════════════════════════════════════════════════════════════════════
def fig_2d_single(theta, LD, VV0, model, sx, sy):
    H_loop, M_loop, Hs, Ms = predict_beta_sweep(model, sx, sy, theta, LD, VV0)
    dx = np.diff(H_loop); dy = np.diff(M_loop)
    arc = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    _, ui = np.unique(arc, return_index=True)
    arc = arc[ui]; Hr = H_loop[ui]; Mr = M_loop[ui]
    af = np.linspace(0, arc[-1], 800)
    try:
        sH = make_interp_spline(arc, Hr, k=3)(af)
        sM = make_interp_spline(arc, Mr, k=3)(af)
    except Exception:
        sH, sM = Hr, Mr

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(sH, sM, color='#7B2D8B', lw=2.2)
    ax.scatter(Hs, Ms, marker='^', s=55, facecolors='none',
               edgecolors='#e74c3c', lw=1.4)
    ax.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.6)
    ax.axvline(0, color='gray', lw=0.7, ls='--', alpha=0.6)
    ax.set_xlabel(r'$H/s_u LD$', fontsize=16)
    ax.set_ylabel(r'$M/s_u L^2 D$', fontsize=16)
    ax.set_title('Failure envelopes of caisson foundations near slope\nunder combined loading',
                 fontweight='bold', fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.text(0.03, 0.97, f'$\\theta$ = {theta}°,  $L/D$ = {LD}\n$V/V_0$ = {VV0}',
            transform=ax.transAxes, ha='left', va='top', fontsize=13,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='black', alpha=0.85))
    fig.tight_layout()
    return fig


def fig_2d_sweep(theta, LD, model, sx, sy):
    VV0_list = [0.0, 0.5, 0.75]
    colors = ['#7B2D8B', '#e74c3c', '#2980b9']
    linestyles = ['-', '--', '-.']
    fig, ax = plt.subplots(figsize=(7.5, 7))
    scaled = predict_all_vv0(model, sx, sy, theta, LD)
    for vv0, color, ls in zip(VV0_list, colors, linestyles):
        H_loop, M_loop = scaled[vv0]
        sH, sM = smooth_loop(H_loop, M_loop)
        ax.plot(sH, sM, color=color, lw=2.2, linestyle=ls, label=f'$V/V_0$ = {vv0}')
        ax.scatter(H_loop[:-1], M_loop[:-1], marker='^', s=45,
                   facecolors='none', edgecolors=color, lw=1.2)
    ax.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.6)
    ax.axvline(0, color='gray', lw=0.7, ls='--', alpha=0.6)
    ax.set_xlabel(r'$H/s_u LD$', fontsize=16)
    ax.set_ylabel(r'$M/s_u L^2 D$', fontsize=16)
    ax.set_title('Failure envelopes of caisson foundations near slope\nunder combined loading',
                 fontweight='bold', fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=12, loc='upper left')
    ax.text(0.97, 0.97, f'$\\theta$ = {theta}°,  $L/D$ = {LD}',
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='black', alpha=0.85))
    fig.tight_layout()
    return fig


def fig_3d(theta, LD, model, sx, sy):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    VV0_list = [0.0, 0.25, 0.5, 0.75]
    scaled = predict_all_vv0(model, sx, sy, theta, LD)
    cmap = plt.colormaps['plasma']; norm_c = Normalize(vmin=0, vmax=0.75)
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection='3d')

    loops = []
    for vv0 in VV0_list:
        H_arr, M_arr = scaled[vv0]
        color = cmap(norm_c(vv0))
        ax.plot(H_arr, M_arr, np.full_like(H_arr, vv0), color=color, lw=2.0, zorder=5)
        ax.scatter(H_arr, M_arr, np.full_like(H_arr, vv0), color=color, s=10, alpha=0.6)
        loops.append((H_arr, M_arr, vv0))

    for i in range(len(loops) - 1):
        H0, M0, v0 = loops[i]; H1, M1, v1 = loops[i + 1]
        n = min(len(H0), len(H1))
        verts = [((H0[j], M0[j], v0), (H0[j + 1], M0[j + 1], v0),
                  (H1[j + 1], M1[j + 1], v1), (H1[j], M1[j], v1)) for j in range(n - 1)]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.18,
                             facecolor=cmap(norm_c((v0 + v1) / 2)), edgecolor='none'))

    H_top, M_top, v_top = loops[-1]; n = len(H_top)
    cap = [((H_top[j], M_top[j], v_top),
            (H_top[(j + 1) % n], M_top[(j + 1) % n], v_top), (0, 0, v_top)) for j in range(n)]
    ax.add_collection3d(Poly3DCollection(cap, alpha=0.25,
                         facecolor=cmap(norm_c(v_top)), edgecolor='none'))

    H_bot, M_bot, v_bot = loops[0]
    bot = [((H_bot[j], M_bot[j], v_bot),
            (H_bot[(j + 1) % n], M_bot[(j + 1) % n], v_bot), (0, 0, v_bot)) for j in range(n)]
    ax.add_collection3d(Poly3DCollection(bot, alpha=0.15,
                         facecolor=cmap(norm_c(0)), edgecolor='none'))

    sm = ScalarMappable(cmap=cmap, norm=norm_c); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(r'$V/V_0$', fontsize=14)
    cbar.set_ticks([0, 0.25, 0.5, 0.75])
    ax.set_xlabel(r'$H/s_u LD$', fontsize=14, labelpad=10)
    ax.set_ylabel(r'$M/s_u L^2 D$', fontsize=14, labelpad=10)
    ax.set_zlabel(r'$V/V_0$', fontsize=14, labelpad=8)
    ax.set_zlim(0, 0.75)
    ax.set_zticks([0.0, 0.25, 0.5, 0.75])
    ax.set_title(f'3D Failure Envelope — Caisson near Slope\n$\\theta$ = {theta}°,  $L/D$ = {LD}',
                fontsize=16, pad=18, fontweight='bold')
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Export table
# ═══════════════════════════════════════════════════════════════════════════════
def export_table_bytes(theta, LD, VV0, model, sx, sy, fmt):
    H_loop, M_loop, _, _ = predict_beta_sweep(model, sx, sy, theta, LD, VV0)
    beta_half = list(range(0, 181, 2))
    beta_full = beta_half + [b + 180 for b in beta_half[1:]]
    df = pd.DataFrame({
        'theta_deg': theta, 'L_D': LD, 'V_V0': VV0,
        'beta_deg': beta_full,
        'H_over_suLD': H_loop,
        'M_over_suL2D': M_loop,
    })
    buf = io.BytesIO()
    if fmt == 'csv':
        buf.write(df.to_csv(index=False).encode('utf-8'))
    else:
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df.to_excel(w, index=False, sheet_name='Prediction')
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title='DeepFM V-H-M Failure Envelope', layout='wide')
setup_fonts()

st.title('V–H–M Failure Envelope of Caisson Foundations near Slope')
st.caption('DeepFM Predictor — Caisson Foundations near Sloping Seabed in Clay')

if not os.path.exists(MODEL_PATH):
    st.error(f'Model file not found: {MODEL_PATH}. '
            f'Upload slope_deepfm_best.pt into the web_app/ folder before deploying.')
    st.stop()

model, sx, sy = load_model()

col_input, col_plot = st.columns([1, 2.2])

with col_input:
    st.subheader('Inputs')
    theta = st.number_input('θ — slope angle (°)', min_value=THETA_RANGE[0],
                            max_value=THETA_RANGE[1], value=30.0, step=1.0)
    LD = st.number_input('L/D', min_value=LD_RANGE[0], max_value=LD_RANGE[1],
                         value=10.0, step=1.0)
    VV0 = st.slider('V/V₀', min_value=VV0_RANGE[0], max_value=VV0_RANGE[1],
                    value=0.0, step=0.05)
    st.caption(f'Training data domain: θ∈[{THETA_RANGE[0]:.0f}°, {THETA_RANGE[1]:.0f}°], '
              f'L/D∈[{LD_RANGE[0]:.0f}, {LD_RANGE[1]:.0f}], '
              f'V/V₀∈[{VV0_RANGE[0]:.2f}, {VV0_RANGE[1]:.2f}] — stay inside this '
              f'range to avoid extrapolation.')

    st.divider()
    st.subheader('Export prediction table')
    fmt = st.radio('Format', ['xlsx', 'csv'], horizontal=True)
    data_bytes = export_table_bytes(theta, LD, VV0, model, sx, sy, fmt)
    st.download_button(
        label=f'Download .{fmt}',
        data=data_bytes,
        file_name=f'prediction_theta{theta:g}_LD{LD:g}_VV0{VV0:g}.{fmt}',
        mime='text/csv' if fmt == 'csv' else
             'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    if os.path.exists(IMG_PATH):
        st.divider()
        st.image(IMG_PATH, caption='Problem definition', width='stretch')

with col_plot:
    tab_2d, tab_sweep, tab_3d = st.tabs(
        ['2D envelope (single V/V₀)', '2D envelope (V/V₀ sweep)', '3D envelope'])
    with tab_2d:
        st.pyplot(fig_2d_single(theta, LD, VV0, model, sx, sy), clear_figure=True)
    with tab_sweep:
        st.pyplot(fig_2d_sweep(theta, LD, model, sx, sy), clear_figure=True)
    with tab_3d:
        st.pyplot(fig_3d(theta, LD, model, sx, sy), clear_figure=True)

st.divider()
st.caption('This is a reviewer-facing demo of the trained DeepFM surrogate model used in the '
          'manuscript. Source code: SlopeGUI.py (desktop GUI) in the accompanying repository.')
