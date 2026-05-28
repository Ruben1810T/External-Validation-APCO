import streamlit as st
import pandas as pd
import numpy as np
import pyvital
import os
import statsmodels.api as sm
from scipy.signal import find_peaks
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

load_dotenv('vars.env')
DATA_PATH   = os.getenv("DATA_PATH")
SAMPLE_LENGTH = int(os.getenv("SAMPLE_LENGTH"))
SAMPLING_RATE = float(os.getenv("SAMPLING_RATE"))

# ── Verwerkingsfuncties ────────────────────────────────────────────────────────

def load_vital(pad):
    df_v = pd.read_csv(pad)
    df_v['Time']    = pd.to_timedelta(df_v['Time'])
    df_v['minutes'] = (df_v['Time'] - df_v['Time'].min()).dt.total_seconds() / 60
    return df_v

def load_hemosphere(pad):
    df_h = pd.read_csv(pad, sep=';')
    df_h = df_h.iloc[1:].reset_index(drop=True)
    df_h['Tijd'] = pd.to_timedelta(df_h['Tijd'].str.strip())
    df_h['SV'] = pd.to_numeric(df_h['SV'].str.strip().replace(r'\\n', np.nan, regex=True), errors='coerce')
    df_h = df_h.dropna(subset=['SV']).reset_index(drop=True)
    return df_h

def resample_abp(df):
    ABP = df[['minutes', 'Intellivue/ABP']].dropna().reset_index(drop=True)
    ABP.columns = ['minutes', 'ABP']
    time_diff   = ABP['minutes'].diff().dropna() * 60
    sample_freq = 1 / time_diff.mean()
    resample_ABP = np.array(pyvital.resample_hz(ABP['ABP'], sample_freq, 100), dtype=np.float32)
    t_start = ABP['minutes'].min()
    return resample_ABP, t_start

def link_abp_sv(df, df_h, resample_ABP):
    time_vital       = df['Time'].dt.total_seconds().values
    time_hs          = df_h['Tijd'].dt.total_seconds().values
    sv_values        = df_h['SV'].values
    t_abp_klok_start = time_vital[0]
    t_abp_klok_end   = time_vital[-1]

    samples_abp, samples_sv, sv_times = [], [], []
    for i in range(len(time_hs)):
        sv_time   = time_hs[i]
        if sv_time < t_abp_klok_start or sv_time > t_abp_klok_end:
            continue
        sv_idx    = round((sv_time - t_abp_klok_start) * 100)
        start_idx = sv_idx - SAMPLE_LENGTH
        if start_idx < 0 or sv_idx > len(resample_ABP):
            continue
        segment = np.array(resample_ABP[start_idx:sv_idx])
        if len(segment) != SAMPLE_LENGTH:
            continue
        samples_abp.append(segment)
        samples_sv.append(sv_values[i])
        sv_times.append(sv_time)

    samples_abp = np.array(samples_abp, dtype=np.float32)
    samples_sv  = np.array(samples_sv)
    sv_times    = np.array(sv_times)
    g_indices   = ((sv_times - t_abp_klok_start) / 20).astype(int)
    return samples_abp, samples_sv, sv_times, g_indices

def lowess_smoothing(samples_sv, indices_sv, sampling_rate=1/100):
    diffs        = np.diff(indices_sv.reshape(-1))
    slicing_locs = (np.where(diffs > 200 / sampling_rate)[0] + 1).tolist()
    slicing_locs = [0] + slicing_locs + [len(indices_sv) + 1]
    samples_sv_smoothed = np.array([])
    for slice_index in range(1, len(slicing_locs)):
        i_slice = indices_sv[slicing_locs[slice_index - 1]:slicing_locs[slice_index]]
        s_slice = samples_sv[slicing_locs[slice_index - 1]:slicing_locs[slice_index]]
        s_slice_smoothed = sm.nonparametric.lowess(
            exog=i_slice.reshape([len(i_slice)]),
            endog=s_slice.reshape([len(s_slice)]),
            frac=0.03, return_sorted=False)
        samples_sv_smoothed = np.append(samples_sv_smoothed, s_slice_smoothed)
    return samples_sv_smoothed.reshape(len(samples_sv_smoothed), 1)

def lowess_sv(samples_sv, g_indices):
    indices_sv = (g_indices * 2000).reshape(-1, 1)
    return lowess_smoothing(samples_sv.reshape(-1, 1), indices_sv, SAMPLING_RATE)

def filter_physiological(segments, sv_smoothed, indices_to_be_removed):
    mask = (
        (segments.min(axis=1) < 25) | (segments.max(axis=1) > 250) |
        (sv_smoothed.flatten() < 20) | (sv_smoothed.flatten() > 200)
    )
    indices_to_be_removed.update(np.where(mask)[0])

def detect_unrealistic_segment(segment, max_step=25, max_noise=5):
    diff = np.abs(np.diff(segment))
    return diff.max() > max_step or diff.std() > max_noise

def filter_noise(segments, indices_to_be_removed, max_step=25, max_noise=5):
    mask = np.array([detect_unrealistic_segment(seg, max_step, max_noise) for seg in segments])
    indices_to_be_removed.update(np.where(mask)[0])

def filter_heartrate(segments, sv_times, indices_to_be_removed):
    for i, seg in enumerate(segments):
        mins, maxs = pyvital.detect_peaks(seg, srate=100)
        mins, maxs = np.array(mins), np.array(maxs)
        if len(mins) < 2 or len(maxs) < 2:
            indices_to_be_removed.add(i)
            continue
        median_interval = np.median(np.concatenate([np.diff(mins), np.diff(maxs)]))
        hr_bpm = 1 / (median_interval * SAMPLING_RATE) * 60
        if np.isnan(hr_bpm) or not (30 <= hr_bpm <= 180):
            indices_to_be_removed.add(i)

def filter_pulse_pressure(segments, indices_to_be_removed):
    for i, seg in enumerate(segments):
        mins, maxs = pyvital.detect_peaks(seg, srate=100)
        mins, maxs = np.array(mins), np.array(maxs)
        if len(mins) < 1 or len(maxs) < 2:
            indices_to_be_removed.add(i)
            continue
        min_vals = seg[mins]
        max_vals = seg[maxs]
        pps = [max_vals[m + 1] - min_vals[m] for m in range(len(min_vals)) if m + 1 < len(max_vals)]
        pp  = np.mean(pps) if pps else np.nan
        if np.isnan(pp) or pp < 20:
            indices_to_be_removed.add(i)

def filter_extra_peaks(segments, indices_to_be_removed, prominence=2, factor=1.2):
    for i, seg in enumerate(segments):
        _, maxs_hs = pyvital.detect_peaks(seg, srate=100)
        if len(maxs_hs) < 2:
            indices_to_be_removed.add(i)
            continue
        n_golven      = len(maxs_hs)
        all_maxs, _   = find_peaks(seg,  prominence=prominence)
        all_mins, _   = find_peaks(-seg, prominence=prominence)
        if len(all_maxs) > n_golven * 3 * factor or len(all_mins) > n_golven * 3 * factor:
            indices_to_be_removed.add(i)

@st.cache_data
def laad_patient(vital_pad, hs_pad):
    df_v         = load_vital(vital_pad)
    df_h         = load_hemosphere(hs_pad)
    resample_ABP, _ = resample_abp(df_v)
    segments, sv, sv_times, g_indices = link_abp_sv(df_v, df_h, resample_ABP)
    sv_smoothed  = lowess_sv(sv, g_indices)
    return segments, sv_smoothed.flatten(), sv_times, g_indices

def bereken_filters(segments, sv_smoothed, sv_times, g_indices, params):
    indices_to_be_removed = set()
    if params['fysiologisch']:
        filter_physiological(segments, sv_smoothed.reshape(-1, 1), indices_to_be_removed)
    if params['ruis']:
        filter_noise(segments, indices_to_be_removed, params['max_step'], params['max_noise'])
    if params['hartslag']:
        filter_heartrate(segments, sv_times, indices_to_be_removed)
    if params['pulse_pressure']:
        filter_pulse_pressure(segments, indices_to_be_removed)
    if params['extra_pieken']:
        filter_extra_peaks(segments, indices_to_be_removed, params['prominence'], params['factor'])
    return indices_to_be_removed

# ── Plotfuncties (Plotly) ──────────────────────────────────────────────────────

def plot_voor_na(segments, sv, indices_to_be_removed, patient_naam):
    mask = np.ones(len(segments), dtype=bool)
    mask[list(indices_to_be_removed)] = False

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Voor filtering", "Na filtering", "SV"],
                        vertical_spacing=0.07)

    for i in range(len(segments)):
        x     = np.arange(i * SAMPLE_LENGTH, (i + 1) * SAMPLE_LENGTH) / 100 / 60
        kleur = "#E24B4A" if not mask[i] else "#85B7EB"
        fig.add_trace(go.Scatter(x=x, y=segments[i], mode='lines',
                                 line=dict(color=kleur, width=0.5),
                                 showlegend=False), row=1, col=1)

    signaal_na = np.full(len(segments) * SAMPLE_LENGTH, np.nan)
    for i in np.where(mask)[0]:
        signaal_na[i * SAMPLE_LENGTH:(i + 1) * SAMPLE_LENGTH] = segments[i]
    x_na = np.arange(len(signaal_na)) / 100 / 60
    fig.add_trace(go.Scatter(x=x_na, y=signaal_na, mode='lines',
                             line=dict(color="#85B7EB", width=0.5),
                             showlegend=False), row=2, col=1)

    x_sv    = np.arange(len(sv)) * SAMPLE_LENGTH / 100 / 60
    kleuren = ["#E24B4A" if not mask[i] else "#85B7EB" for i in range(len(sv))]
    fig.add_trace(go.Scatter(x=x_sv, y=sv, mode='markers',
                             marker=dict(color=kleuren, size=4),
                             showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_sv[mask], y=sv[mask], mode='lines',
                             line=dict(color="#378ADD", width=1),
                             showlegend=False), row=3, col=1)

    fig.update_yaxes(title_text="ABP (mmHg)", row=1, col=1)
    fig.update_yaxes(title_text="ABP (mmHg)", row=2, col=1)
    fig.update_yaxes(title_text="SV (mL)",    row=3, col=1)
    fig.update_xaxes(title_text="Tijd (min)", row=3, col=1)
    fig.update_layout(title=patient_naam, height=800)
    return fig

def plot_hartslag(segments, sv_times, indices_to_be_removed):
    hr_list, hr_times = [], []
    t_offset = sv_times[0] / 60 - 20 / 60

    for i, seg in enumerate(segments):
        mins, maxs = pyvital.detect_peaks(seg, srate=100)
        mins, maxs = np.array(mins), np.array(maxs)
        if len(mins) >= 2 and len(maxs) >= 2:
            median_interval = np.median(np.concatenate([np.diff(mins), np.diff(maxs)]))
            hr_bpm = 1 / (median_interval * SAMPLING_RATE) * 60
            if not np.isnan(hr_bpm):
                hr_list.append(hr_bpm)
                hr_times.append(sv_times[i] / 60 - t_offset)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hr_times, y=hr_list, mode='lines+markers',
                             line=dict(color="#378ADD"), marker=dict(size=4)))
    fig.add_hline(y=30,  line_dash="dash", line_color="#E24B4A", annotation_text="30 bpm")
    fig.add_hline(y=180, line_dash="dash", line_color="#E24B4A", annotation_text="180 bpm")
    fig.update_layout(title="Hartslag over tijd", xaxis_title="Tijd (min)",
                      yaxis_title="Hartslag (bpm)", height=400)
    return fig

def plot_pulse_pressure(segments, sv_times, indices_to_be_removed):
    pp_list, pp_times = [], []
    t_offset = sv_times[0] / 60 - 20 / 60

    for i, seg in enumerate(segments):
        mins, maxs = pyvital.detect_peaks(seg, srate=100)
        mins, maxs = np.array(mins), np.array(maxs)
        if len(mins) >= 1 and len(maxs) >= 2:
            min_vals = seg[mins]
            max_vals = seg[maxs]
            pps = [max_vals[m + 1] - min_vals[m] for m in range(len(min_vals)) if m + 1 < len(max_vals)]
            pp  = np.mean(pps) if pps else np.nan
        else:
            pp = np.nan
        pp_list.append(pp)
        pp_times.append(sv_times[i] / 60 - t_offset)

    pp_array = np.array(pp_list)
    kleuren  = ["#E24B4A" if i in indices_to_be_removed else "#378ADD" for i in range(len(segments))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pp_times, y=pp_array, mode='markers',
                             marker=dict(color=kleuren, size=5)))
    fig.add_hline(y=20, line_dash="dash", line_color="#E24B4A", annotation_text="20 mmHg")
    fig.update_layout(title="Pulse pressure over tijd", xaxis_title="Tijd (min)",
                      yaxis_title="Pulse pressure (mmHg)", height=400)
    return fig

def plot_pieken(segments, prominence):
    signaal = segments.flatten()
    x       = np.arange(len(signaal)) / 100 / 60
    maxs, _ = find_peaks(signaal,  prominence=prominence)
    mins, _ = find_peaks(-signaal, prominence=prominence)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=signaal, mode='lines',
                             line=dict(color="#85B7EB", width=0.4), name="Signaal"))
    fig.add_trace(go.Scatter(x=x[maxs], y=signaal[maxs], mode='markers',
                             marker=dict(color="#E24B4A", size=4), name="Piek"))
    fig.add_trace(go.Scatter(x=x[mins], y=signaal[mins], mode='markers',
                             marker=dict(color="#E8A838", size=4), name="Dal"))
    fig.update_layout(title="Piekdetectie heel signaal", xaxis_title="Tijd (min)",
                      yaxis_title="ABP (mmHg)", height=400)
    return fig

# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="ABP Preprocessing", layout="wide")
st.title("ABP Preprocessing Dashboard")

# Patiënten laden
vital_base = os.path.join(DATA_PATH, 'Vital data')
hs_base    = os.path.join(DATA_PATH, 'HemoSphere data')
patienten  = {}
for folder in sorted(os.listdir(vital_base)):
    vital_pad = os.path.join(vital_base, folder, 'vital.csv')
    hs_pad    = os.path.join(hs_base,    folder, 'hemosphere.csv')
    if os.path.exists(vital_pad) and os.path.exists(hs_pad):
        patient_id = folder.replace('MARTINI_', '')
        patienten[patient_id] = (vital_pad, hs_pad)

# Sidebar
with st.sidebar:
    st.header("Instellingen")

    patient_keuze = st.selectbox("Patiënt", list(patienten.keys()))

    st.subheader("Filters aan/uit")
    f_fysiologisch  = st.checkbox("Fysiologisch",   value=True)
    f_ruis          = st.checkbox("Ruis",            value=True)
    f_hartslag      = st.checkbox("Hartslag",        value=True)
    f_pp            = st.checkbox("Pulse pressure",  value=True)
    f_extra_pieken  = st.checkbox("Extra pieken",    value=True)

    st.subheader("Parameters")
    max_step   = st.slider("max_step (mmHg)",   10, 60,  25)
    max_noise  = st.slider("max_noise (std)",    1, 20,   5)
    prominence = st.slider("prominence (mmHg)", 1, 10,   2)
    factor     = st.slider("factor pieken",    1.0, 3.0, 1.2)

# Data laden en filteren
vital_pad, hs_pad = patienten[patient_keuze]
with st.spinner("Data laden..."):
    segments, sv_smoothed, sv_times, g_indices = laad_patient(vital_pad, hs_pad)

params = {
    'fysiologisch':  f_fysiologisch,
    'ruis':          f_ruis,
    'hartslag':      f_hartslag,
    'pulse_pressure': f_pp,
    'extra_pieken':  f_extra_pieken,
    'max_step':      max_step,
    'max_noise':     max_noise,
    'prominence':    prominence,
    'factor':        factor,
}

with st.spinner("Filters berekenen..."):
    indices_to_be_removed = bereken_filters(segments, sv_smoothed, sv_times, g_indices, params)

mask = np.ones(len(segments), dtype=bool)
mask[list(indices_to_be_removed)] = False

col1, col2, col3 = st.columns(3)
col1.metric("Totaal segmenten", len(segments))
col2.metric("Na filtering",     int(mask.sum()))
col3.metric("Verwijderd",       len(indices_to_be_removed))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Voor / Na filtering", "Hartslag", "Pulse pressure", "Pieken"])

with tab1:
    st.plotly_chart(plot_voor_na(segments, sv_smoothed, indices_to_be_removed, patient_keuze),
                    use_container_width=True)

with tab2:
    st.plotly_chart(plot_hartslag(segments, sv_times, indices_to_be_removed),
                    use_container_width=True)

with tab3:
    st.plotly_chart(plot_pulse_pressure(segments, sv_times, indices_to_be_removed),
                    use_container_width=True)

with tab4:
    st.plotly_chart(plot_pieken(segments, prominence),
                    use_container_width=True)
