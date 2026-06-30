"""
Interface graphique Streamlit : visualise une partie jouée par l'agent Eagle-1.

Lancer avec :
    streamlit run gui/app.py

Prérequis : l'API doit tourner en parallèle (uvicorn api.main:app --port 8000)
"""

import time

import gymnasium as gym
import numpy as np
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Eagle-1 Autopilot", page_icon="🚀", layout="wide")

st.title("🚀 Eagle-1 — Pilote automatique d'atterrissage lunaire")
st.caption("Visualisation d'une partie jouée par l'agent DQN entraîné")

# ---------- Sidebar : vérif API + paramètres ----------
with st.sidebar:
    st.header("Paramètres")

    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API en ligne ({health['status']})")
    except requests.exceptions.RequestException:
        st.error(f"API injoignable sur {API_URL}")
        st.stop()

    seed = st.number_input("Seed (graine aléatoire)", value=42, min_value=0, max_value=10000, step=1)
    fps = st.slider("Vitesse (fps)", min_value=10, max_value=120, value=50)
    st.divider()

    if st.button("🛬 Lancer un atterrissage", type="primary", use_container_width=True):
        st.session_state["run"] = True
    else:
        st.session_state.setdefault("run", False)


# ---------- Exécution d'un épisode ----------
if st.session_state.get("run"):
    with st.spinner("Récupération de la trajectoire via l'API..."):
        resp = requests.post(f"{API_URL}/play", json={"seed": int(seed)}, timeout=30).json()

    trajectory = resp["trajectory"]
    total_reward = resp["total_reward"]
    n_steps = resp["steps"]
    success = resp["success"]
    soft_landing = resp["soft_landing"]

    # Métriques en haut
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Récompense totale", f"{total_reward:.1f}", delta="✅ Réussi" if success else "❌ Échoué")
    c2.metric("Étapes", n_steps)
    c3.metric("Atterrissage doux", "Oui" if soft_landing else "Non")
    c4.metric("Seed", seed)

    st.divider()

    col_render, col_state = st.columns([2, 1])

    # ---------- Rejouer l'épisode localement pour l'animation ----------
    # On rejoue avec la même seed + actions de la trajectoire pour obtenir les frames.
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env.reset(seed=int(seed))

    with col_render:
        st.subheader("Animation de l'atterrissage")
        frame_slot = st.empty()

    with col_state:
        st.subheader("État courant")
        info_slot = st.empty()
        step_progress = st.progress(0.0)

    obs_labels = ["x", "y", "vx", "vy", "angle", "vit. angulaire", "jambe G", "jambe D"]
    delay = 1.0 / fps

    for i, log in enumerate(trajectory):
        action = log["action"]
        env.step(action)
        frame = env.render()

        frame_slot.image(frame, channels="RGB", use_container_width=True)

        state = log["state"]
        action_names = ["rien", "← gauche", "↑ principal", "→ droit"]
        info_slot.markdown(
            f"**Étape :** {log['step']}/{n_steps}  \n"
            f"**Action :** {action_names[action]}  \n"
            f"**Reward :** {log['reward']:.2f}  \n\n"
            "**Observations :**\n"
            + "\n".join(f"- {lbl} : `{val:+.3f}`" for lbl, val in zip(obs_labels, state))
        )
        step_progress.progress((i + 1) / n_steps)
        time.sleep(delay)

    env.close()
    st.success("Atterrissage terminé ✈️")

else:
    st.info("Configure une seed dans la barre latérale et clique sur **Lancer un atterrissage**.")