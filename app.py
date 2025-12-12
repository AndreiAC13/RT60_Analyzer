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
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False
if 'rt_active' not in st.session_state: st.session_state.rt_active = False
if 'stipa_active' not in st.session_state: st.session_state.stipa_active = False

# --- HEADER ---
st.title("🎛️ Raumakustik Auswertungstool")
st.markdown("##### Erstellt von HSMW Student Andrei Zamshev")
st.markdown("Analyse nach **DIN 18041 (Gruppe A & B)** und **DIN EN IEC 60268-16**")


# --- TABS ---
tab1, tab2 = st.tabs(["📊 RT60/EDT/C50/C80/D50 (DIN 18041)", "🗣️ STIPA"])

# --- HILFSFUNKTIONEN ---

def get_param_info(param_name):
    """Liefert detaillierte Erklärungen."""
    info = {
        "T30": """
        **Nachhallzeit T30**
        Definition: T30 misst den Abfall von 30 dB (T30 = 2 x Zeit für 30 dB Abfall), während die RT60 die volle Zeit bis zum 60 dB Abfall angibt (RT60 = 3 x T20, oder T30 x 2).
        In der Praxis wird oft T20 oder T30 gemessen, da das Signal-Rausch-Verhältnis für 60 dB Abfall oft nicht ausreicht.
        """,
        "T20": """
        **Nachhallzeit T20**
        Analog zu T30, aber berechnet über einen Abfall von 20 dB.
        """,
        "EDT": """
        **Early Decay Time (EDT)**
        Beschreibt den Abfall der ersten 10 dB. Wichtig für subjektiven Eindruck.
        """,
        "C50": """
        **Klarheitsmaß C50 (Sprache)**
        Zielwert: > 0 dB (besser > +3 dB).
        """,
        "C80": """
        **Klarheitsmaß C80 (Musik)**
        Zielwert für Musik: -2 bis +2 dB.
        """,
        "D50": """
        **Deutlichkeit D50**
        Zielwert für gute Sprachverständlichkeit: > 50 %.
        """
    }
    for key, text in info.items():
        if key in param_name: return text
    return ""

def calculate_target_AV_ratio(usage, h):
    """
    Berechnet das erforderliche A/V Verhältnis nach DIN 18041 Tabelle 3
    für Räume der Gruppe B.
    """
    if "B1" in usage:
        return 0.0 # Keine Anforderung
    
    # Bestimmung der Basisparameter je nach Nutzung
    # Werte aus Tabelle 3 der Norm
    #-----#GRUPPE B BERECHNUNG#-----#
    #-----#GRUPPE B BERECHNUNG#-----#
    #-----#GRUPPE B BERECHNUNG#-----#
    #-----#GRUPPE B BERECHNUNG#-----#
    #-----#GRUPPE B BERECHNUNG#-----#
    #-----#GRUPPE B BERECHNUNG#-----#
    limits = {
        "B2": {"small": 0.15, "x": 4.80},
        "B3": {"small": 0.20, "x": 3.13},
        "B4": {"small": 0.25, "x": 2.13},
        "B5": {"small": 0.30, "x": 1.47}
    }
    
    key = usage[:2] # z.B. "B3"
    if key not in limits: return 0.0
    
    val = limits[key]
    
    result = 0.0
    if h <= 2.5:
        # Spalte "bei Raumhöhen h <= 2,5 m"
        result = val["small"]
    else:
        # Formel für h > 2.5m: A/V >= [x + 4.69 * lg(h)]^-1
        denom = val["x"] + 4.69 * np.log10(h)
        result = 1.0 / denom
        
    return round(result, 2)  #gerundet

def create_pdf(v_raum, cat, plot_fig, df_res, param_label, av_results=None):
    """Erstellt PDF Bericht inkl. Gruppe B Daten falls vorhanden."""
    pdf = FPDF()
    
    # Seite 1: Grafik
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Bericht: {param_label}", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Raum: {v_raum} m3 | Nutzung: {cat}", 0, 1, 'C')
    
    if av_results and "B" in cat:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Ergebnisse DIN 18041 Gruppe B:", 0, 1, 'C')
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"Soll A/V: > {av_results['target']:.2f} 1/m", 0, 1, 'C')
        pdf.cell(0, 8, f"Ist A/V: {av_results['actual']:.2f} 1/m", 0, 1, 'C')
        status = "ERFÜLLT" if av_results['passed'] else "NICHT ERFÜLLT"
        pdf.set_text_color(0, 128, 0) if av_results['passed'] else pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 8, f"Status: {status}", 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plot_fig.savefig(tmpfile.name, dpi=150, bbox_inches='tight')
        pdf.image(tmpfile.name, x=10, y=60 if av_results else 40, w=190)
    
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
# ==============================================================================
# TAB 1: RT60 & PARAMETER & GRUPPE B
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

        V = st.number_input("Volumen V (m³)", value=226.0, step=1.0)
        
        # Erweiterte Nutzungsarten inkl. Gruppe B (letzte bearbeitung 12.12.25)
        nutzungsarten = [
            "A1 - Musik", 
            "A2 - Sprache/Vortrag",
            "A3 - Unterricht / Sprache inklusiv", 
            "A4 - Unterricht inklusiv", 
            "A5 - Sport",
            "B1 - ohne Aufenthaltsqualität",
            "B2 - kurzfristiges Verweilen",
            "B3 - längerfristiges Verweilen",
            "B4 - Lärmminderung & Komfort",
            "B5 - besonderer Lärmschutz"
        ]
        cat = st.selectbox("Nutzung (DIN 18041):", nutzungsarten)

        # --- LOGIK FÜR GRUPPE A (Soll-Nachhallzeit) ---
        # --- LOGIK FÜR GRUPPE A (Soll-Nachhallzeit) ---
        # --- LOGIK FÜR GRUPPE A (Soll-Nachhallzeit) ---
        # --- LOGIK FÜR GRUPPE A (Soll-Nachhallzeit) ---
        # --- LOGIK FÜR GRUPPE A (Soll-Nachhallzeit) ---
        t_soll = 0.0 # Startwert 0 bedeutet: Keine Linie zeichnen
        is_group_b = "B" in cat.split(" - ")[0]

        if cp["rt"] and not is_group_b:
            if "A1" in cat: t_soll = 0.45 * np.log10(V) + 0.07 if 30<=V<=1000 else st.error("A1: Volumen muss zwischen 30 und 1000 m³ liegen.")
            elif "A2" in cat:
                if 50 <= V < 5000: t_soll = 0.37 * np.log10(V) - 0.14
                else: st.error("A2: Volumen muss zwischen 50 und 5000 m³ liegen.")
            elif "A3" in cat: t_soll = 0.32 * np.log10(V) - 0.17 if 30<=V<5000 else st.error("A3: Volumen muss zwischen 30 und 5000 m³ liegen.")
            elif "A4" in cat: t_soll = 0.26 * np.log10(V) - 0.14 if 30<=V<500 else st.error("A4: Volumen muss zwischen 30 und 500 m³ liegen.")
            elif "A5" in cat:
                # A5: ab 200 m3
                if 200 <= V < 10000:
                    t_soll = 0.75 * np.log10(V) - 1.00
                elif V >= 10000:
                    t_soll = 2.0
                else:
                    st.error("A5: Volumen muss mindestens 200 m³ betragen.")
                    # Ergebnis anzeigen (nur wenn gültig)
            if t_soll > 0:
                t_soll = round(t_soll, 2)
                st.success(f"Ziel Tₘ: {t_soll} s")

        # --- LOGIK FÜR GRUPPE B (Zusätzliche Eingabefelder + Sabine Rückrechnung) ---
         # --- LOGIK FÜR GRUPPE B (Zusätzliche Eingabefelder + Sabine Rückrechnung) ---
          # --- LOGIK FÜR GRUPPE B (Zusätzliche Eingabefelder + Sabine Rückrechnung) ---
           # --- LOGIK FÜR GRUPPE B (Zusätzliche Eingabefelder + Sabine Rückrechnung) ---
           
        target_AV = 0.0
        h_raum = 0.0
        a_calc_input = 0.0
        
        if is_group_b:
            st.markdown("---")
            st.markdown("**Parameter für Gruppe B:**")
            h_raum = st.number_input("Raumhöhe h (m)", value=3.0, step=0.1, help="Lichte Raumhöhe für DIN 18041 Tab. 3")
            
            # Sollwert Berechnung A/V
            target_AV = calculate_target_AV_ratio(cat, h_raum)
            
            if target_AV > 0:
                st.info(f"Soll A/V-Verhältnis: ≥ {target_AV:.2f} $m^{-1}$")
                
                # RÜCKRECHNUNG AUF NACHHALLZEIT VIA SABINE
                # Sabine: T = 0.163 * V / A
                # Da A = V * (A/V), kürzt sich V raus:
                # T = 0.163 / (A/V)
                if cp["rt"]:
                    t_soll = 0.163 / target_AV
                    t_soll = round(t_soll, 2)
                    st.success(f"Ziel T (Sabine): ≤ {t_soll} s")
            else:
                st.info("Keine Anforderung an A/V (B1)")

            a_calc_input = st.number_input("Äquiv. Absorptionsfläche A (m²)", value=0.0, step=1.0, 
                                           help="Manuelle Eingabe überschreibt die Berechnung aus Messdateien.")

        # Farben
        with st.expander("🎨 Farben"):
            c_single = st.color_picker("Einzelmessungen", "#cccccc", key="c1")
            c_mean   = st.color_picker("Mittelwert", "#000000", key="c2")
            if cp["rt"]:
                c_zone   = st.color_picker("Toleranzbereich / Zielkorridor", "#e6f5e6", key="c3")
                c_target = st.color_picker("Sollwert-Linie", "#4CAF50", key="c4")

        st.markdown("---")
        if st.button("🗑️ Reset"):
            st.session_state.confirm_reset = True
        
        if st.session_state.confirm_reset:
            if st.button("Bestätigen"):
                st.session_state.uploader_key += 1
                st.session_state.rt_active = False 
                st.session_state.confirm_reset = False
                st.rerun()

    with c_main:
        files = st.file_uploader(f"Dateien für {cp['label']}", accept_multiple_files=True, type="txt", key=f"u_{st.session_state.uploader_key}")
        
        # Variablen initialisieren für PDF Export später
        av_results_pdf = None
        
        # Trigger
        trigger_calc = st.button("🚀 Auswerten", type="primary", use_container_width=True)
        
        if trigger_calc:
            st.session_state.rt_active = True
        
        if st.session_state.rt_active:
            # --- DATENVERARBEITUNG (falls Dateien vorhanden) ---
            freqs = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
            res, warns = [], []
            calculated_A_mean = 0.0
            
            if files:
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
            
            # --- BERECHNUNGEN UND PLOT ---
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
                    
                    # Für Gruppe B: Berechnung von A aus T im Bereich 250Hz - 2000Hz aus den Dateien
                    if is_group_b:
                        indices_b = [np.where(freqs==f)[0][0] for f in [250, 500, 1000, 2000] if f in freqs]
                        if indices_b:
                            t_vals_b = [mean_val[i] for i in indices_b]
                            # Sabine Formel: A = 0.163 * V / T
                            a_vals = [(0.163 * V / t) for t in t_vals_b if t > 0]
                            if a_vals:
                                calculated_A_mean = np.mean(a_vals)

                # --- PLOT ---
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Toleranzschlauch für Gruppe A ODER Gruppe B (wenn t_soll berechnet wurde)
                if cp["rt"] and t_soll > 0:
                    tf_def = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
                    # Für Gruppe B nutzen wir dieselben Toleranzkurven als Orientierung wie A
                    uf = [1.7, 1.45, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2] if "A5" not in cat else [1.7, 1.45, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
                    lf = [0.5, 0.65, 0.8, 0.8, 0.8, 0.8, 0.65, 0.5] if "A5" not in cat else [0.5, 0.65, 0.8, 0.8, 0.8, 0.8, 0.65, 0.5]
                    
                    t_u = np.interp(freqs, tf_def, uf) * t_soll
                    t_l = np.interp(freqs, tf_def, lf) * t_soll
                    
                    label_tol = 'DIN 18041 Toleranz' if not is_group_b else 'Zielkorridor (Orientierung)'
                    ax.fill_between(freqs, t_l, t_u, color=c_zone, label=label_tol)
                    ax.axhline(t_soll, color=c_target, linestyle='--', linewidth=2, label=f'Zielwert {t_soll}s')

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
                
                # PDF Vorbereitung (Diagramm ist da)
                plot_fig_ref = fig
                df_ex_ref = pd.DataFrame({"Freq": freqs, "Mittelwert": mean_val, "StdAbw": std_val})
                if cp["rt"] and t_soll>0: 
                    df_ex_ref["Min"], df_ex_ref["Max"] = t_l, t_u
            
            else:
                # Fallback wenn KEINE Dateien, aber vielleicht manuell A eingegeben für Gruppe B
                if is_group_b and a_calc_input > 0:
                    st.info("Keine Messdateien hochgeladen. Berechnung basiert nur auf manueller Eingabe.")
                    # Dummy Plot für PDF
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, "Keine Messdaten\nNur manuelle A/V Berechnung", ha='center')
                    plot_fig_ref = fig
                    df_ex_ref = pd.DataFrame({"Info": ["Keine Messdaten"]})
                    tm = 0; fs = 0
                    mat = np.zeros((1,1)) # Dummy
                else:
                    if not files and not (is_group_b and a_calc_input > 0):
                        st.warning("Bitte Dateien hochladen oder (für Gruppe B) Parameter manuell eingeben.")
                        st.stop()

            # --- METRIKEN & AUSWERTUNG ---
            c1, c2, c3 = st.columns(3)
            
            if is_group_b and cp["rt"]:
                # GRUPPE B AUSWERTUNG
                st.markdown("### 🏛️ Auswertung DIN 18041 Gruppe B")
                
                # LOGIKÄNDERUNG: Manuelle Eingabe hat Vorrang vor berechnetem Wert
                if a_calc_input > 0:
                    final_A = a_calc_input
                    source_msg = "(Manuelle Eingabe)"
                else:
                    final_A = calculated_A_mean
                    source_msg = "(Berechnet aus T)"

                if final_A > 0:
                    actual_AV = final_A / V
                    passed = actual_AV >= target_AV
                    
                    av_results_pdf = {
                        "target": target_AV,
                        "actual": actual_AV,
                        "passed": passed
                    }

                    k1, k2, k3 = st.columns(3)
                    k1.metric(f"Verwendetes A {source_msg}", f"{final_A:.1f} m²")
                    k2.metric("Ist A/V Verhältnis", f"{actual_AV:.2f} 1/m")
                    k3.metric("Soll A/V (Tab. 3)", f"≥ {target_AV:.2f} 1/m", delta="Erfüllt" if passed else "Nicht Erfüllt", delta_color="normal" if passed else "inverse")
                else:
                    st.warning("Konnte A nicht ermitteln. Bitte A manuell eingeben oder T-Messungen hochladen.")
            else:
                # GRUPPE A METRIKEN (nur wenn Messdaten da sind)
                if 'res' in locals() and res:
                    if cp["rt"]:
                        c1.metric("T_mid (gemessen)", f"{tm:.2f} s")
                        c2.metric("Schröder-Freq.", f"{fs:.0f} Hz")
                    else:
                        v_mid = np.nanmean([mean_val[i500], mean_val[i1k]])
                        c1.metric("Mittelwert (500-1k)", f"{v_mid:.2f} {cp['unit']}")
                    c3.metric("Anzahl Messungen", f"{mat.shape[1]}")

            # --- EXPORT ---
            col1, col2 = st.columns(2)
            
            if 'df_ex_ref' in locals():
                b_ex = io.BytesIO()
                with pd.ExcelWriter(b_ex, engine='openpyxl') as w: df_ex_ref.to_excel(w, index=False)
                col1.download_button("📊 Excel Export", b_ex.getvalue(), "daten.xlsx", use_container_width=True)
                
                pdf_d = create_pdf(V, cat, plot_fig_ref, df_ex_ref, cp['label'], av_results=av_results_pdf)
                col2.download_button("📄 PDF Bericht", pdf_d, "bericht.pdf", "application/pdf", use_container_width=True)

                with st.expander("📋 Tabelle"):
                    st.dataframe(df_ex_ref.style.format("{:.2f}") if "Freq" in df_ex_ref else df_ex_ref, use_container_width=True, hide_index=True)

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