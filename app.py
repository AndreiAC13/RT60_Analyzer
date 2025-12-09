import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re
from fpdf import FPDF
import tempfile

# --- 1. KONFIGURATION ---
st.set_page_config(page_title="Raumakustik Pro", layout="wide", initial_sidebar_state="collapsed")

# --- SESSION STATE INITIALISIERUNG ---
# Wir speichern hier den Status, ob eine Berechnung erfolgt ist
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False
if 'rt_active' not in st.session_state: st.session_state.rt_active = False
if 'stipa_active' not in st.session_state: st.session_state.stipa_active = False

# --- HILFSFUNKTIONEN ---

def get_param_info(param_name):
    """Liefert detaillierte Erklärungen."""
    info = {
        "T30": """
        **Nachhallzeit T30**
        Die Zeitspanne, in der der Schalldruckpegel um 60 dB abfällt (berechnet aus dem Abfall von -5 bis -35 dB). 
        Maßgeblich für DIN 18041.
        """,
        "T20": """
        **Nachhallzeit T20**
        Analog zu T30, aber berechnet über einen Abfall von 20 dB (-5 bis -25 dB). 
        Nützlich in lauten Räumen, falls T30 nicht messbar ist.
        """,
        "EDT": """
        **Early Decay Time (EDT)**
        Beschreibt den Abfall der ersten 10 dB. Entspricht oft besser dem subjektiven Halleindruck als T30.
        """,
        "C50": """
        **Klarheitsmaß C50 (Sprache)**
        Verhältnis der Schallenergie der ersten 50 ms zur restlichen Energie. 
        Zielwert: > 0 dB (besser > +3 dB).
        """,
        "C80": """
        **Klarheitsmaß C80 (Musik)**
        Verhältnis der Schallenergie der ersten 80 ms zur restlichen Energie. 
        Zielwert für Musik: -2 bis +2 dB.
        """,
        "D50": """
        **Deutlichkeit D50**
        Prozentualer Anteil der Schallenergie der ersten 50 ms. 
        Zielwert für gute Sprachverständlichkeit: > 50 %.
        """
    }
    for key, text in info.items():
        if key in param_name: return text
    return ""

def create_pdf(v_raum, cat, plot_fig, df_res, param_label):
    """Erstellt PDF Bericht."""
    pdf = FPDF()
    
    # Seite 1: Grafik
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Bericht: {param_label}", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Raum: {v_raum} m3 | Nutzung: {cat}", 0, 1, 'C')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plot_fig.savefig(tmpfile.name, dpi=150, bbox_inches='tight')
        pdf.image(tmpfile.name, x=10, y=40, w=190)
    
    # Seite 2: Tabelle
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Messergebnisse", 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 9)
    col_w = 25 if len(df_res.columns) > 5 else 30
    for col in df_res.columns:
        pdf.cell(col_w, 8, str(col), 1, 0, 'C')
    pdf.ln()
    
    pdf.set_font("Arial", '', 9)
    for i, row in df_res.iterrows():
        for item in row:
            val = str(item)
            try: val = f"{float(item):.2f}"
            except: pass
            pdf.cell(col_w, 7, val, 1, 0, 'C')
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

# --- HEADER ---
st.title("🎛️ Raumakustik Auswertungstool")
st.markdown("##### Erstellt von HSMW Student Andrei Zamshev")
st.markdown("Analyse nach **DIN 18041** und **DIN EN IEC 60268-16**")

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Raumakustik (RT60)", "🗣️ STIPA"])

# ==============================================================================
# TAB 1: RT60 & PARAMETER
# ==============================================================================
with tab1:
    c_set, c_main = st.columns([1, 3])
    
    with c_set:
        st.subheader("⚙️ Einstellungen")
        
        # Parameter Wahl
        param_choice = st.selectbox("Parameter:", 
            ["T30 (Nachhall)", "T20 (Nachhall)", "EDT", "C50 (Sprache)", "C80 (Musik)", "D50 (Deutlichkeit)"])
        
        param_map = {
            "T30": {"idx": 6, "r": 7, "unit": "s", "label": "T30", "rt": True},
            "T20": {"idx": 4, "r": 5, "unit": "s", "label": "T20", "rt": True},
            "EDT": {"idx": 2, "r": 3, "unit": "s", "label": "EDT", "rt": True},
            "C50": {"idx": 14, "r": None, "unit": "dB", "label": "C50", "rt": False},
            "C80": {"idx": 15, "r": None, "unit": "dB", "label": "C80", "rt": False},
            "D50": {"idx": 16, "r": None, "unit": "%", "label": "D50", "rt": False}
        }
        key = next(k for k in param_map if k in param_choice)
        cp = param_map[key] 

        with st.expander("ℹ️ Info"): st.markdown(get_param_info(key))

        V = st.number_input("Volumen (m³)", value=226.0, step=1.0)
        cat = st.selectbox("Nutzung:", ["A1 - Musik", "A3 - Unterricht", "A4 - Inklusion", "A5 - Sport"])
        
        t_soll = 0.0
        if cp["rt"]:
            if "A1" in cat: t_soll = 0.45 * np.log10(V) + 0.07 if 30<=V<=30000 else 1.0
            elif "A3" in cat: t_soll = 0.32 * np.log10(V) - 0.17 if 30<=V<5000 else 0.0
            elif "A4" in cat: t_soll = 0.26 * np.log10(V) - 0.14 if 30<=V<5000 else 0.0
            elif "A5" in cat: 
                t_soll = 0.75 * np.log10(V) - 1.00 if 30<=V<10000 else 2.0
                if t_soll < 1.3 and V > 200: t_soll = 1.3
            t_soll = round(t_soll, 2)
            if t_soll > 0: st.success(f"Ziel Tₘ: {t_soll} s")
            else: st.warning("Volumen ungeeignet")

        # Farben
        with st.expander("🎨 Farben"):
            c_single = st.color_picker("Einzelmessungen", "#cccccc", key="c1")
            c_mean   = st.color_picker("Mittelwert", "#000000", key="c2")
            if cp["rt"]:
                c_zone   = st.color_picker("Toleranzbereich", "#e6f5e6", key="c3")
                c_target = st.color_picker("Sollwert-Linie", "#4CAF50", key="c4")

        st.markdown("---")
        if st.button("🗑️ Reset"):
            st.session_state.confirm_reset = True
        
        if st.session_state.confirm_reset:
            if st.button("Bestätigen"):
                st.session_state.uploader_key += 1
                st.session_state.rt_active = False # Reset active state
                st.session_state.confirm_reset = False
                st.rerun()

    with c_main:
        files = st.file_uploader(f"Dateien für {cp['label']}", accept_multiple_files=True, type="txt", key=f"u_{st.session_state.uploader_key}")
        
        if files:
            # Wenn Button gedrückt wird, merken wir uns den Status
            if st.button("🚀 Auswerten", type="primary", use_container_width=True):
                st.session_state.rt_active = True
            
            # Überprüfen, ob Auswertung aktiv ist
            if st.session_state.rt_active:
                freqs = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
                res, warns = [], []
                
                for f in files:
                    try:
                        c = f.getvalue().decode("utf-8")
                        s = io.StringIO(c)
                        fd = []
                        bad_r = 0
                        for l in s:
                            if l.strip() and l.strip()[0].isdigit():
                                p = l.split(',')
                                if len(p) >= 17:
                                    fr = float(p[0])
                                    v = float(p[cp["idx"]])
                                    if cp["r"] and 125 <= fr <= 4000:
                                        try:
                                            if abs(float(p[cp["r"]])) < 0.95: bad_r += 1
                                        except: pass
                                    fd.append([fr, v])
                        if fd:
                            if bad_r > 2: warns.append(f.name)
                            a = np.array(fd)
                            row = []
                            for tf in freqs:
                                idx = (np.abs(a[:,0]-tf)).argmin()
                                if abs(a[idx,0]-tf) < 5:
                                    val = a[idx, 1]
                                    if cp["rt"] and (val<=0 or val>15): row.append(np.nan)
                                    else: row.append(val)
                                else: row.append(np.nan)
                            res.append(row)
                    except: pass
                
                if res:
                    if warns: st.warning(f"⚠️ Unsichere Messungen (r < 0.95): {', '.join(warns)}")
                    
                    mat = np.array(res).T
                    mean_val = np.nanmean(mat, axis=1)
                    std_val = np.nanstd(mat, axis=1)
                    
                    i500 = np.where(freqs==500)[0][0]
                    i1k = np.where(freqs==1000)[0][0]
                    tm = 0; fs = 0
                    if cp["rt"]:
                        tm = np.nanmean([mean_val[i500], mean_val[i1k]])
                        if tm > 0: fs = 2000 * np.sqrt(tm/V)

                    # --- PLOT ---
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if cp["rt"] and t_soll > 0:
                        tf_def = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
                        uf = [1.7, 1.45, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2] if "A5" not in cat else [1.7, 1.45, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
                        lf = [0.5, 0.65, 0.8, 0.8, 0.8, 0.8, 0.65, 0.5] if "A5" not in cat else [0.5, 0.65, 0.8, 0.8, 0.8, 0.8, 0.65, 0.5]
                        
                        t_u = np.interp(freqs, tf_def, uf) * t_soll
                        t_l = np.interp(freqs, tf_def, lf) * t_soll
                        
                        ax.fill_between(freqs, t_l, t_u, color=c_zone, label='DIN 18041 Toleranz')
                        ax.axhline(t_soll, color=c_target, linestyle='--', linewidth=2, label=f'Sollwert {t_soll}s')

                    if fs > 0 and cp["rt"]:
                        ax.axvline(fs, color='blue', linestyle=':', label=f'Schröder ({int(fs)} Hz)')

                    for i in range(mat.shape[1]):
                        ax.semilogx(freqs, mat[:, i], color=c_single, alpha=0.5, linewidth=0.8)

                    ax.errorbar(freqs, mean_val, yerr=std_val, fmt='o-', color=c_mean, 
                                linewidth=2.5, markersize=6, capsize=4, label='Mittelwert & StdAbw')

                    ax.set_xscale('log')
                    ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000, 8000])
                    ax.set_xticklabels(['63', '125', '250', '500', '1k', '2k', '4k', '8k'])
                    ax.set_xlim(50, 10000)
                    if cp["rt"]: ax.set_ylim(bottom=0)
                    
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    ax.set_xlabel("Frequenz [Hz]", fontsize=12)
                    ax.set_ylabel(f"{cp['label']} [{cp['unit']}]", fontsize=12)
                    ax.set_title(f"{cp['label']} Analyse - {cat} (V={int(V)} m³)", fontsize=14)
                    ax.legend(loc='upper right', framealpha=0.9)
                    
                    st.pyplot(fig)

                    c1, c2, c3 = st.columns(3)
                    if cp["rt"]:
                        c1.metric("T_mid (gemessen)", f"{tm:.2f} s")
                        c2.metric("Schröder-Freq.", f"{fs:.0f} Hz")
                    else:
                        v_mid = np.nanmean([mean_val[i500], mean_val[i1k]])
                        c1.metric("Mittelwert (500-1k)", f"{v_mid:.2f} {cp['unit']}")
                    c3.metric("Anzahl Messungen", f"{mat.shape[1]}")

                    col1, col2 = st.columns(2)
                    
                    df_ex = pd.DataFrame({"Freq": freqs, "Mittelwert": mean_val, "StdAbw": std_val})
                    if cp["rt"] and t_soll>0: df_ex["Min"], df_ex["Max"] = t_l, t_u
                    
                    b_ex = io.BytesIO()
                    with pd.ExcelWriter(b_ex, engine='openpyxl') as w: df_ex.to_excel(w, index=False)
                    col1.download_button("📊 Excel Export", b_ex.getvalue(), "daten.xlsx", use_container_width=True)
                    
                    pdf_d = create_pdf(V, cat, fig, df_ex, cp['label'])
                    col2.download_button("📄 PDF Bericht", pdf_d, "bericht.pdf", "application/pdf", use_container_width=True)

                    with st.expander("📋 Tabelle"):
                        st.dataframe(df_ex.style.format("{:.2f}"), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: STIPA
# ==============================================================================
with tab2:
    st.subheader("STIPA Auswertung")
    c_s1, c_s2 = st.columns([1, 3])
    
    with c_s1:
        st.write("Grenzwerte:")
        lim_g = st.slider("Gut", 0.4, 0.8, 0.60)
        lim_f = st.slider("Akzeptabel", 0.3, 0.7, 0.50)
        grp = st.number_input("Gruppierung (n Dateien)", 1, 10, 1)
        if st.button("🗑️ Reset", key="rst_s"): 
            st.session_state.uploader_key += 1
            st.session_state.stipa_active = False
            st.rerun()

    with c_s2:
        s_files = st.file_uploader("NTi Dateien", accept_multiple_files=True, key=f"s_{st.session_state.uploader_key}")
        
        if s_files:
            if st.button("Starten", type="primary", key="go_s"):
                st.session_state.stipa_active = True
            
            if st.session_state.stipa_active:
                raw = []
                fs_sort = sorted(s_files, key=lambda x: x.name)
                for f in fs_sort:
                    try:
                        txt = f.getvalue().decode("utf-8", errors='ignore')
                        m = re.findall(r'\d{2}:\d{2}:\d{2}\s+(0,\d{2})', txt)
                        if m: raw.append(float(m[-1].replace(',', '.')))
                    except: pass
                
                if raw:
                    ng = int(np.ceil(len(raw)/grp))
                    g_v, g_n = [], []
                    for i in range(ng):
                        c = raw[i*grp : (i+1)*grp]
                        g_v.append(np.mean(c))
                        g_n.append(f"Pos {i+1}")
                    
                    fig_s, ax_s = plt.subplots(figsize=(10, 5))
                    
                    ax_s.axhspan(0, lim_f, color='#ffcccc', alpha=0.5, label='Schlecht')
                    ax_s.axhspan(lim_f, lim_g, color='#ffffcc', alpha=0.5, label='Akzeptabel')
                    ax_s.axhspan(lim_g, 1.0, color='#ccffcc', alpha=0.5, label='Gut')
                    
                    bars = ax_s.bar(g_n, g_v, color='#333333', zorder=3)
                    
                    for bar in bars:
                        h = bar.get_height()
                        ax_s.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", 
                                  ha='center', va='bottom', fontweight='bold')

                    ax_s.set_ylim(0, 1.0)
                    ax_s.set_ylabel("STIPA Index")
                    ax_s.set_title(f"STIPA Ergebnisse (n={grp} gemittelt)")
                    ax_s.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
                    ax_s.axhline(lim_g, color='green', linestyle='--', linewidth=2)
                    
                    st.pyplot(fig_s)
                    
                    df_s = pd.DataFrame({"Pos": g_n, "Wert": g_v})
                    bs = io.BytesIO()
                    with pd.ExcelWriter(bs, engine='openpyxl') as w: df_s.to_excel(w, index=False)
                    st.download_button("📊 Excel", bs.getvalue(), "stipa.xlsx")