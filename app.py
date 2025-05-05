import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go

def load_spectrum(f):
    content = f.read().decode('utf-8', errors='ignore').splitlines()
    data = []
    for line in content:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                wl, val = float(parts[0]), float(parts[1])
                data.append((wl, val))
            except:
                continue
    arr = np.array(data)
    return (arr[:,0], arr[:,1]) if arr.size else (np.array([]), np.array([]))

def calc_metrics(y1, y2):
    m = {
        'Pearson': pearsonr(y1, y2)[0],
        'Spearman': spearmanr(y1, y2)[0]
    }
    norm1, norm2 = np.linalg.norm(y1), np.linalg.norm(y2)
    m['Cosine'] = float(np.dot(y1, y2)/(norm1*norm2)) if norm1 and norm2 else 0.0
    angle = np.arccos(np.clip(np.dot(y1, y2)/(norm1*norm2), -1,1)) if norm1 and norm2 else 0.0
    m['SAM (deg)'] = float(np.degrees(angle))
    diff = y1 - y2
    m['RMSE'] = float(np.sqrt(np.mean(diff**2)))
    max_pat = max(np.max(y1),1e-6)
    m['RMSE relativo vs Patrón'] = m['RMSE']/max_pat
    return m

def sim_total(m):
    cos, sam = m['Cosine'], m['SAM (deg)']
    rr = m['RMSE relativo vs Patrón']
    pr, sr = m['Pearson'], m['Spearman']
    return (0.45*cos + 0.25*max(0,1-sam/180) + 0.15*max(0,1-rr) +
            0.10*((pr+1)/2) + 0.05*((sr+1)/2)) * 100

st.title("Comparador Espectral")
modo = st.sidebar.selectbox("Modo de datos", ["Reflectancia", "Transmitancia"])
pat_files = st.sidebar.file_uploader("Patrones (.txt,.csv)", type=['txt','csv'], accept_multiple_files=True)
sam_files = st.sidebar.file_uploader("Muestras (.txt,.csv)", type=['txt','csv'], accept_multiple_files=True)

if pat_files and sam_files:
    # load & align
    pats = [load_spectrum(f) for f in pat_files]
    base_wl, _ = pats[0]
    aligned_p = [np.interp(base_wl, wl, y) if wl.shape!=base_wl.shape or not np.allclose(wl, base_wl) else y
                 for wl, y in pats]
    rep_pat = np.mean(aligned_p, axis=0)
    sams = [load_spectrum(f) for f in sam_files]
    aligned_sam = [np.interp(base_wl, wl, y) if wl.shape!=base_wl.shape or not np.allclose(wl, base_wl) else y
                   for wl, y in sams]
    rep_sam = np.mean(aligned_sam, axis=0)

    if rep_pat.size<2 or rep_sam.size<2:
        st.error("Datos insuficientes.")
        st.stop()

    # metrics
    m = calc_metrics(rep_pat, rep_sam)
    total = sim_total(m)
    m['Similitud total (%)'] = total
    st.subheader("Métricas de similitud")
    st.table(pd.DataFrame.from_dict(m, orient='index', columns=['Valor']).round(4))

    # detect top 3 per band
    bands = [(350,1400),(1400,2000),(2000,2250)]
    feats = []
    for start,end in bands:
        mask = (base_wl>=start)&(base_wl<=end)
        wl_band = base_wl[mask]; pat_band = rep_pat[mask]; sam_band = rep_sam[mask]
        if wl_band.size<3:
            continue
        prom = 0.05*(np.max(pat_band)-np.min(pat_band))
        if modo=="Transmitancia":
            idx, props = signal.find_peaks(-pat_band, prominence=prom, distance=1)
            vals = -pat_band[idx]
        else:
            idx, props = signal.find_peaks(pat_band, prominence=prom, distance=1)
            vals = pat_band[idx]
        if idx.size==0:
            continue
        # select top 3
        N = min(3, idx.size)
        order = np.argsort(props["prominences"])[-N:]
        for i in order:
            pos = idx[i]
            feats.append((wl_band[pos], pat_band[pos], sam_band[pos]))

    # sort by wavelength
    feats = sorted(feats, key=lambda x: x[0])
    df_feats = pd.DataFrame(feats, columns=["Wavelength (nm)", f"Valor ({modo} Patrón)", f"Valor ({modo} Muestra)"]).round(4)
    st.subheader("Top 3 características por banda")
    st.table(df_feats)

    # plot
    color_pat, color_sam = "#1f77b4","#ff7f0e"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_wl,y=rep_pat,mode="lines",name="Patrón",line=dict(color=color_pat,width=3)))
    fig.add_trace(go.Scatter(x=base_wl,y=rep_sam,mode="lines",name="Muestra",line=dict(color=color_sam,width=3,dash="dash")))
    ymin,ymax = min(rep_pat.min(),rep_sam.min()), max(rep_pat.max(),rep_sam.max())
    for wl_val,_,_ in feats:
        fig.add_shape(type="line",x0=wl_val,y0=ymin,x1=wl_val,y1=ymax,line=dict(color="gray",dash="dash"))
    fig.update_traces(hovertemplate="λ: %{x:.2f} nm<br>Valor: %{y:.4f}")
    fig.update_layout(title=f"Símil: {total:.2f}%",xaxis_title="Longitud de onda (nm)",yaxis_title=modo,hovermode="x unified")
    st.subheader("Gráfico interactivo")
    st.plotly_chart(fig,use_container_width=True)
