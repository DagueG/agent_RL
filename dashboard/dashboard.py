"""
Tableau de bord Streamlit : suivi interactif des performances de l'agent Eagle-1.

Lancer avec :
    streamlit run dashboard/dashboard.py

Source de données : data/eval_logs.csv (généré par notebooks/log_eval_episodes.py)
"""

import pandas as pd
import streamlit as st

CSV_PATH = "data/eval_logs.csv"

st.set_page_config(page_title="Eagle-1 — Dashboard", page_icon="📊", layout="wide")
st.title("📊 Eagle-1 — Tableau de bord des performances")
st.caption("Analyse des 100 épisodes d'évaluation du meilleur modèle DQN")

# ---------- Chargement ----------
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"Fichier introuvable : {CSV_PATH}. Lance d'abord `python notebooks/log_eval_episodes.py`.")
    st.stop()

# ---------- Sidebar : filtres ----------
with st.sidebar:
    st.header("Filtres")

    ep_min, ep_max = int(df["episode"].min()), int(df["episode"].max())
    ep_range = st.slider("Plage d'épisodes", ep_min, ep_max, (ep_min, ep_max))

    outcome = st.multiselect(
        "Résultat",
        options=["Succès (≥200)", "Échec (<200)"],
        default=["Succès (≥200)", "Échec (<200)"],
    )

    landing = st.multiselect(
        "Type d'atterrissage",
        options=["Doux", "Brutal"],
        default=["Doux", "Brutal"],
    )

# ---------- Application des filtres ----------
filt = df[(df["episode"] >= ep_range[0]) & (df["episode"] <= ep_range[1])].copy()

if "Succès (≥200)" not in outcome:
    filt = filt[filt["success"] != 1]
if "Échec (<200)" not in outcome:
    filt = filt[filt["success"] != 0]
if "Doux" not in landing:
    filt = filt[filt["soft_landing"] != 1]
if "Brutal" not in landing:
    filt = filt[filt["soft_landing"] != 0]

if filt.empty:
    st.warning("Aucun épisode ne correspond aux filtres.")
    st.stop()

# ---------- KPIs ----------
st.subheader("Indicateurs clés")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Épisodes", len(filt))
c2.metric("Reward moyen", f"{filt['reward'].mean():.1f}")
c3.metric("Écart-type", f"{filt['reward'].std():.1f}")
c4.metric("Taux de succès", f"{filt['success'].mean() * 100:.0f}%")
c5.metric("Atterrissages doux", f"{filt['soft_landing'].mean() * 100:.0f}%")

st.divider()

# ---------- Graphiques ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Récompense par épisode")
    chart_data = filt.set_index("episode")[["reward"]].copy()
    chart_data["seuil 200"] = 200
    st.line_chart(chart_data)

with col2:
    st.subheader("Moyenne mobile (fenêtre = 10)")
    smoothed = filt.set_index("episode")[["reward"]].rolling(window=10, min_periods=1).mean()
    smoothed.columns = ["reward (moyenne mobile)"]
    st.line_chart(smoothed)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribution des récompenses")
    # Histogramme via st.bar_chart
    bins = pd.cut(filt["reward"], bins=10)
    hist = bins.value_counts().sort_index()
    hist.index = [f"{int(iv.left)}/{int(iv.right)}" for iv in hist.index]
    st.bar_chart(hist)

with col4:
    st.subheader("Longueur des épisodes")
    st.bar_chart(filt.set_index("episode")[["steps"]])

st.divider()

# ---------- Répartition succès / échec ----------
st.subheader("Répartition succès / échec")
c_repart1, c_repart2 = st.columns(2)
with c_repart1:
    success_count = filt["success"].value_counts().rename({1: "Succès", 0: "Échec"})
    st.bar_chart(success_count)
with c_repart2:
    landing_count = filt["soft_landing"].value_counts().rename({1: "Doux", 0: "Brutal"})
    st.bar_chart(landing_count)

# ---------- Position finale ----------
st.subheader("Position finale d'atterrissage")
st.caption("La zone cible est centrée en x=0. Plus le point est proche de zéro, mieux c'est.")
final_pos = filt[["final_x", "final_y", "success"]].copy()
final_pos.columns = ["x", "y", "success"]
st.scatter_chart(final_pos, x="x", y="y", color="success")

# ---------- Données brutes ----------
with st.expander("Voir les données brutes filtrées"):
    st.dataframe(filt, use_container_width=True)