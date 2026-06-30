# Eagle-1 — Pilote automatique d'atterrissage lunaire

Projet de formation OpenClassrooms — Ingénieur Machine Learning, spécialité Apprentissage par Renforcement.

Agent **DQN** (Stable-Baselines3) entraîné à piloter le module Eagle-1 dans l'environnement Gymnasium **LunarLander-v3**, avec API d'inférence (FastAPI), interface graphique (Streamlit) et tableau de bord interactif (Streamlit).

## Performance atteinte

| Métrique | Cible | Obtenu |
|---|---|---|
| Récompense moyenne sur 100 épisodes | ≥ 200 | **~270** ✅ |
| Écart-type | faible | **~37** |
| Atterrissages réussis (reward ≥ 200) | — | **96/100** |
| Atterrissages doux (deux jambes au sol) | — | **85/100** |

## Structure du projet

```
agent_RL/
├── notebooks/
│   └── mission_eagle1.ipynb        # notebook principal de la mission
├── exercice3_dqn_cartpole.ipynb    # notebook des exercices 1-3 (DQN CartPole)
├── scripts/                        # scripts utilitaires (régénération vidéo, CSV, etc.)
├── api/main.py                     # API FastAPI : /health, /info, /predict, /play
├── gui/app.py                      # Interface graphique Streamlit
├── dashboard/dashboard.py          # Tableau de bord Streamlit
├── models/best_model.zip           # Meilleur modèle entraîné
├── data/eval_logs.csv              # Logs des 100 épisodes d'évaluation
├── videos/eagle1_demo.mp4          # Démo vidéo de 22.7s
├── logs/                           # Logs TensorBoard
├── requirements.txt
└── README.md
```

## Installation

Prérequis : Python 3.11+ (Python 3.9 ne fournit pas de wheel précompilé pour `box2d-py` sous Windows).

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sous Windows, si l'installation de `gymnasium[box2d]` échoue :

```bash
pip install swig
pip install "gymnasium[box2d]"
```

## Utilisation

### Notebook principal

```bash
jupyter notebook notebooks/mission_eagle1.ipynb
```

Le notebook contient l'intégralité de la démarche : exploration, choix d'algorithme, baseline, optimisation des hyperparamètres, évaluation finale, et formatage des données pour l'API.

**Important** : exécuter depuis la racine du projet pour que les chemins relatifs (`models/`, `data/`, `logs/`) fonctionnent.

### API (FastAPI)

```bash
uvicorn api.main:app --port 8000
```

Documentation interactive sur http://localhost:8000/docs

Endpoints :
- `GET /health` — vérification de la disponibilité de l'API
- `GET /info` — métadonnées du modèle et de l'environnement
- `POST /predict` — état (8 floats) → action (0-3)
- `POST /play` — joue un épisode complet, renvoie trajectoire + métriques

### Interface graphique (Streamlit)

Nécessite que l'API tourne en parallèle (voir ci-dessus).

```bash
streamlit run gui/app.py
```

Permet de choisir une seed, de lancer un atterrissage et de visualiser l'animation frame par frame avec l'état courant de l'agent.

### Tableau de bord (Streamlit)

Indépendant de l'API, consomme uniquement `data/eval_logs.csv`.

```bash
streamlit run dashboard/dashboard.py
```

KPIs, courbes de récompense, distribution, position d'atterrissage, avec filtres interactifs (plage d'épisodes, succès/échec, type d'atterrissage).

### Régénérer les artefacts

Tous les scripts utilitaires sont dans `scripts/` :

```bash
python scripts/log_eval_episodes.py   # régénère data/eval_logs.csv
python scripts/record_video.py        # régénère des épisodes vidéo
python scripts/concat_videos.py       # concatène en videos/eagle1_demo.mp4
```

## Choix techniques

- **DQN** : adapté à l'espace d'action discret (4 actions) de LunarLander-v3.
- **Hyperparamètres RL Zoo** : `lr=6.3e-4`, `batch_size=128`, `buffer_size=50000`, `target_update_interval=250`, `net_arch=[256, 256]`. Choix justifié dans le notebook.
- **EvalCallback** : sauvegarde automatique du meilleur modèle pendant l'entraînement → contre le *catastrophic forgetting* du DQN.
- **FastAPI** : doc Swagger auto, validation Pydantic, séparation claire backend / frontend.
- **Streamlit** : 100% local pour respecter la contrainte "prototype local sans dépendance à des serveurs externes" du brief.

## Livrables

- ✅ Notebook documentant la démarche complète
- ✅ Vidéo (.mp4, 22.7s) montrant un atterrissage réussi
- ✅ API d'inférence (FastAPI)
- ✅ Interface graphique de visualisation (Streamlit)
- ✅ Tableau de bord interactif (Streamlit)

## Environnement testé

- Python 3.11
- gymnasium 1.1.1
- stable-baselines3 (dernière version)
- PyTorch (CPU)
- Windows 11 / Linux
