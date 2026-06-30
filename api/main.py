"""
API FastAPI pour exposer l'agent DQN entraîné sur LunarLander-v3.

Endpoints :
    GET  /health          -> status check
    GET  /info            -> métadonnées sur le modèle et l'environnement
    POST /predict         -> état (8 floats) -> action (0-3)
    POST /play            -> joue un épisode complet, renvoie trajectoire + métriques

Lancer avec :
    uvicorn api.main:app --reload --port 8000
"""

from typing import List, Optional

import gymnasium as gym
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from stable_baselines3 import DQN

# ---------- Chargement du modèle (une seule fois au démarrage) ----------
MODEL_PATH = "models/best_model.zip"
model = DQN.load(MODEL_PATH)

# Mapping action -> description lisible
ACTION_LABELS = {
    0: "rien",
    1: "propulseur gauche",
    2: "propulseur principal",
    3: "propulseur droit",
}

# Mapping index obs -> description (LunarLander-v3)
OBS_LABELS = [
    "x", "y", "vx", "vy",
    "angle", "vitesse_angulaire",
    "contact_jambe_gauche", "contact_jambe_droite",
]

app = FastAPI(
    title="Eagle-1 Autopilot API",
    description="API d'inférence pour l'agent DQN LunarLander-v3",
    version="1.0.0",
)


# ---------- Schémas Pydantic ----------
class StateRequest(BaseModel):
    """État d'entrée : 8 valeurs continues."""
    state: List[float] = Field(..., min_length=8, max_length=8, description="8 floats LunarLander")
    deterministic: bool = True


class PredictResponse(BaseModel):
    action: int
    action_label: str


class PlayRequest(BaseModel):
    seed: Optional[int] = None
    max_steps: int = 1000


class StepLog(BaseModel):
    step: int
    state: List[float]
    action: int
    reward: float


class PlayResponse(BaseModel):
    total_reward: float
    steps: int
    success: bool
    soft_landing: bool
    trajectory: List[StepLog]


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/info")
def info():
    return {
        "model_path": MODEL_PATH,
        "algorithm": "DQN",
        "environment": "LunarLander-v3",
        "observation_size": 8,
        "observation_labels": OBS_LABELS,
        "action_size": 4,
        "action_labels": ACTION_LABELS,
        "success_threshold": 200,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: StateRequest):
    """Renvoie l'action choisie par l'agent pour un état donné."""
    try:
        obs = np.array(req.state, dtype=np.float32)
        action, _ = model.predict(obs, deterministic=req.deterministic)
        action = int(action)
        return PredictResponse(action=action, action_label=ACTION_LABELS[action])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {e}")


@app.post("/play", response_model=PlayResponse)
def play(req: PlayRequest):
    """Joue un épisode complet côté serveur, renvoie la trajectoire."""
    env = gym.make("LunarLander-v3")
    obs, _ = env.reset(seed=req.seed)

    trajectory: List[StepLog] = []
    total_reward = 0.0
    terminated = truncated = False
    step = 0
    last_obs = obs

    while not (terminated or truncated) and step < req.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        trajectory.append(StepLog(
            step=step,
            state=[float(x) for x in obs],
            action=action,
            reward=float(reward),
        ))

        obs = next_obs
        last_obs = next_obs
        total_reward += float(reward)
        step += 1

    env.close()

    success = total_reward >= 200
    soft_landing = bool(last_obs[6]) and bool(last_obs[7])

    return PlayResponse(
        total_reward=round(total_reward, 2),
        steps=step,
        success=success,
        soft_landing=soft_landing,
        trajectory=trajectory,
    )