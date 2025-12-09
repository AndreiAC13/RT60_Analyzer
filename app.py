import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re

# --- 1. Konfiguration ---
st.set_page_config(page_title="Raumakustik Analyse", layout="wide", initial_sidebar_state="collapsed")

# Session State Init
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False

# --- HEADER & CREDITS ---
st.title("🎛️ Raumakustik Auswertungstool")
st.markdown("##### Erstellt von HSMW Student Andrei Zamshev")
st.markdown("Analyse nach **DIN 18041** und **DIN EN IEC 60268-16**")

# --- TABS erstellen ---
tab1, tab2 = st.tabs(["📊 RT60 (Nachhall)", "🗣️ STIPA (Sprachverständlichkeit)"])

# ==============================================================================
# TAB 1: RT60 ANALYSE
# ==============================================================================
with tab1:
    
    col_settings, col_content = st.columns([1, 3])
    
    # --- LINKE SPALTE: EINSTELLUNGEN ---
    with col_settings:
        st.subheader("⚙️ Einstellungen")
        
        # 1. Raumdaten
        V = st.number_input("Raumvolumen (V) in m³", value=226.0, step=1.0, min_value=10.0, key="v_rt60")
        category = st.selectbox(
            "Nutzungsart (DIN 18041):",
            ["A1 - Musik", "A3 - Unterricht / Kommunikation", "A4 - Unterricht / Kommunikation (inklusiv)", "A5 - Sport"],
            key="cat_rt60"
        )
        
        # Berechnung T_soll
        if "A1" in category:
            t_soll = 0.45 * np.log10(V) + 0.07 if 30 <= V <= 30000 else 1.0
        elif "A3" in category:
            t_soll = 0.32 * np.log10(V) - 0.17 if 30 <= V < 5000 else 0.0
        elif "A4" in category:
            t_soll = 0.26 * np.log10(V) - 0.14 if 30 <= V < 5000 else 0.0
        elif "A5" in category:
            t_soll = 0.75 * np.log10(V) - 1.00 if 30 <= V < 10000 else 2.0
            if t_soll < 1.3 and V > 200: t_soll = 1.3
        
        t_soll = round(t_soll, 2)
        
        if t_soll > 0:
            st.info(f"Zielwert Tₘ: **{t_soll} s**")
        else:
            st.error("Volumen ungeeignet.")

        # 2. Farben
        with st.expander("🎨 Farben & Grafik"):
            c_single = st.color_picker("Einzelmessungen", "#cccccc", key="c1")
            c_mean   = st.color_picker("Mittelwert & StdAbw", "#000000", key="c2")
            c_zone   = st.color_picker("Toleranzbereich", "#e6f5e6", key="c3")
            c_target = st.color_picker("Sollwert-Linie", "#4CAF50", key="c4")

        # 3. RESET BUTTON
        st.markdown("---")
        if st.button("🗑️ Alles zurücksetzen", use_container_width=True):
            st.session_state.confirm_reset = True

        if st.session_state.confirm_reset:
            st.warning("Alle Dateien entfernen?")
            c_yes, c_no = st.columns(2)
            if c_yes.button("Ja", key="yes_rt"):
                st.session_state.uploader_key += 1
                st.session_state.confirm_reset = False
                st.rerun()
            if c_no.button("Nein", key="no_rt"):
                st.session_state.confirm_reset = False
                st.rerun()

    # --- RECHTE SPALTE: INHALT ---
    with col_content:
        rt_files = st.file_uploader(
            "REW Dateien (.txt) hochladen", 
            accept_multiple_files=True, 
            type="txt", 
            key=f"rt_up_{st.session_state.uploader_key}"
        )
        
        if rt_files:
            if st.button("🚀 RT60 Auswerten", key="btn_rt", type="primary"):
                target_freqs = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
                res_mat = []
                
                # Parsing
                for f in rt_files:
                    try:
                        s_io = io.StringIO(f.getvalue().decode("utf-8"))
                        lines = s_io.readlines()
                        f_data = []
                        for line in lines:
                            if line.strip() and line.strip()[0].isdigit():
                                parts = line.split(',')
                                if len(parts) >= 7:
                                    try:
                                        fr = float(parts[0])
                                        val = float(parts[6])
                                        f_data.append([fr, val])
                                    except: continue
                        if not f_data: continue
                        
                        arr = np.array(f_data)
                        row = []
                        for tf in target_freqs:
                            idx = (np.abs(arr[:,0] - tf)).argmin()
                            if abs(arr[idx,0] - tf) < 5: # Toleranz erhöht auf 5 Hz
                                v = arr[idx, 1]
                                row.append(v if 0 < v < 10 else np.nan)
                            else:
                                row.append(np.nan)
                        res_mat.append(row)
                    except: pass
                
                if res_mat:
                    mat_np = np.array(res_mat).T
                    mean_t = np.nanmean(mat_np, axis=1)
                    std_t  = np.nanstd(mat_np, axis=1)
                    
                    # Schröder
                    idx_500, idx_1k = np.where(target_freqs == 500)[0][0], np.where(target_freqs == 1000)[0][0]
                    tm = np.nanmean([mean_t[idx_500], mean_t[idx_1k]])
                    fs = 2000 * np.sqrt(tm/V) if tm > 0 else 0
                    
                    # --- PLOT ---
                    fig, ax = plt.subplots(figsize=(10, 6)) # Optimiertes Format
                    
                    # Toleranz
                    tol_freqs_def = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
                    uf = [1.7, 1.45, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2] if "A5" not in category else [1.7, 1.45, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
                    lf = [0.5, 0.65, 0.8, 0.8, 0.8, 0.8, 0.65, 0.5] if "A5" not in category else [0.5, 0.65, 0.8, 0.8, 0.8, 0.8, 0.65, 0.5]
                    
                    t_upper = np.interp(target_freqs, tol_freqs_def, uf) * t_soll
                    t_lower = np.interp(target_freqs, tol_freqs_def, lf) * t_soll
                    
                    # 1. Toleranz
                    ax.fill_between(target_freqs, t_lower, t_upper, color=c_zone, label='DIN 18041 Toleranz')
                    
                    # 2. Sollwert
                    ax.axhline(t_soll, color=c_target, linestyle='--', linewidth=2, label=f'Sollwert ({t_soll} s)')
                    
                    # 3. Schröder
                    if fs > 0: ax.axvline(fs, color='blue', linestyle=':', label=f'Schröder ({fs:.0f} Hz)')
                    
                    # 4. Einzelmessungen
                    for i in range(mat_np.shape[1]):
                        ax.semilogx(target_freqs, mat_np[:,i], color=c_single, alpha=0.5, linewidth=0.8)
                    
                    # 5. Fehlerbalken
                    ax.errorbar(
                        target_freqs, mean_t, 
                        yerr=std_t, 
                        fmt='none', 
                        ecolor=c_mean, 
                        elinewidth=1.0,
                        capsize=3,
                        capthick=1.0,
                        label='Standardabweichung',
                        zorder=4
                    )

                    # 6. Mittelwert
                    ax.semilogx(
                        target_freqs, mean_t, 
                        color=c_mean, 
                        marker='o', 
                        linewidth=2.0, 
                        markersize=5, 
                        label='Mittelwert',
                        zorder=5
                    )
                    
                    # Achsen & Layout
                    ax.set_xscale('log')
                    ax.set_xlim(50, 10000) # Festgelegt: Start bei 50 Hz
                    ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000, 8000])
                    ax.set_xticklabels(['63', '125', '250', '500', '1k', '2k', '4k', '8k'])
                    ax.set_xlabel("Frequenz [Hz]")
                    ax.set_ylabel("T30 [s]")
                    ax.set_title(f"RT60 - {category} (V={V:.0f} m³)")
                    
                    ax.legend(loc='upper right', framealpha=0.9)
                    ax.grid(True, which="both", linestyle='--', alpha=0.4)
                    ax.set_ylim(bottom=0)
                    
                    st.pyplot(fig, use_container_width=True)
                    
                    # Statistik & Download
                    c1, c2, c3 = st.columns(3)
                    c1.metric("T_mid (gemessen)", f"{tm:.2f} s")
                    c2.metric("Schröder-Freq.", f"{fs:.0f} Hz")
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    c3.download_button("📥 Grafik speichern", buf, "rt60.png", "image/png")

                    # TABELLE (WIEDER HINZUGEFÜGT)
                    st.markdown("### Detaillierte Ergebnisse")
                    df_res = pd.DataFrame({
                        "Frequenz (Hz)": target_freqs,
                        "Mittelwert (s)": np.round(mean_t, 2),
                        "Std.Abw (s)": np.round(std_t, 2),
                        "DIN Min (s)": np.round(t_lower, 2),
                        "DIN Max (s)": np.round(t_upper, 2)
                    })
                    # Formatierung für bessere Lesbarkeit
                    st.dataframe(
                        df_res.style.format("{:.2f}"), 
                        use_container_width=True,
                        hide_index=True
                    )

# ==============================================================================
# TAB 2: STIPA ANALYSE
# ==============================================================================
with tab2:
    
    col_s_set, col_s_con = st.columns([1, 3])
    
    with col_s_set:
        st.subheader("⚙️ Einstellungen")
        stipa_limit = st.slider("Grenze 'Gut'", 0.40, 0.80, 0.60, 0.01)
        stipa_fair = st.slider("Grenze 'Akzeptabel'", 0.30, 0.70, 0.50, 0.01)
        grouping = st.number_input("Gruppierung (Mittelwert aus n Dateien)", 1, 10, 1)
        
        st.markdown("---")
        if st.button("🗑️ Reset STIPA", key="reset_stipa"):
             st.session_state.uploader_key += 1
             st.rerun()

    with col_s_con:
        stipa_files = st.file_uploader(
            "NTi Report Dateien hochladen", 
            accept_multiple_files=True, 
            type="txt", 
            key=f"stipa_up_{st.session_state.uploader_key}"
        )
        
        if stipa_files:
            if st.button("🚀 STIPA Auswerten", key="btn_sti", type="primary"):
                stipa_data = []
                filenames = []
                sorted_files = sorted(stipa_files, key=lambda x: x.name)
                
                for f in sorted_files:
                    try:
                        content = f.getvalue().decode("utf-8", errors='ignore')
                        match = re.findall(r'\d{2}:\d{2}:\d{2}\s+(0,\d{2})', content)
                        if match:
                            val = float(match[-1].replace(',', '.'))
                            stipa_data.append(val)
                            filenames.append(f.name)
                        else:
                            match_alt = re.search(r'STIPA\s.*?(\d[.,]\d{2})', content)
                            if match_alt:
                                val = float(match_alt.group(1).replace(',', '.'))
                                stipa_data.append(val)
                                filenames.append(f.name)
                    except: pass

                if stipa_data:
                    num_groups = int(np.ceil(len(stipa_data) / grouping))
                    g_vals, g_names, g_stds = [], [], []
                    
                    for i in range(num_groups):
                        s, e = i*grouping, min((i+1)*grouping, len(stipa_data))
                        chunk = stipa_data[s:e]
                        g_vals.append(np.mean(chunk))
                        g_stds.append(np.std(chunk))
                        g_names.append(f"Pos {i+1}")

                    # Plot
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    x_lims = [0, num_groups + 1]
                    ax2.fill_between(x_lims, 0, stipa_fair, color='#ffcccc', alpha=0.5, label='Schlecht')
                    ax2.fill_between(x_lims, stipa_fair, stipa_limit, color='#ffffcc', alpha=0.5, label='Akzeptabel')
                    ax2.fill_between(x_lims, stipa_limit, 1.0, color='#ccffcc', alpha=0.5, label='Gut')
                    
                    x_pos = np.arange(1, num_groups + 1)
                    bars = ax2.bar(x_pos, g_vals, yerr=g_stds, capsize=5, color='#333333', alpha=0.8)
                    
                    for bar in bars:
                        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                                f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold')

                    ax2.set_xlim(0.5, num_groups + 0.5)
                    ax2.set_ylim(0, 1.0)
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels(g_names)
                    ax2.set_ylabel("STIPA")
                    ax2.set_title(f"STIPA (n={grouping})")
                    ax2.axhline(stipa_limit, color='green', linestyle='--')
                    
                    st.pyplot(fig2, use_container_width=True)
                    
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
                    buf2.seek(0)
                    st.download_button("📥 Grafik speichern", buf2, "stipa.png", "image/png")

                    # TABELLE (WIEDER HINZUGEFÜGT)
                    st.markdown("### Einzelergebnisse")
                    df_stipa = pd.DataFrame({
                        "Position": g_names,
                        "Mittelwert": np.round(g_vals, 2),
                        "Std.Abw": np.round(g_stds, 3)
                    })
                    st.dataframe(df_stipa, use_container_width=True, hide_index=True)