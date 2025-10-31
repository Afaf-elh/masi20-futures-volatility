# MASI20 Futures Volatility – Prévision

Projet GitHub-first (sans setup local) pour explorer et modéliser la volatilité des contrats à terme sur l’indice MASI20.

## Objectifs
- Construire des séries de rendements et des mesures de volatilité (réalisée, GARCH).
- Tester des modèles GARCH/EGARCH/GJR et comparer leurs performances.
- Préparer une base pour étendre vers la prévision de volatilité implicite et le pricing simple de futures.

## Structure
data/ # raw & processed (non versionnés par défaut)
notebooks/ # analyses exploratoires et modèles
src/ # code réutilisable (features, data, models, evaluate)
models/ # artefacts de modèles (résultats, params)
reports/figures/ # graphiques finaux


## Démarrage (100% GitHub)
1. Ouvre **Codespaces** (bouton vert ⟶ “Create codespace on main”) ou édite via l’UI.
2. Place tes données dans `data/raw/` (ou connecte une source).
3. Ouvre `notebooks/01_exploration_masi20.ipynb` et exécute.

## Environnements
Installe les dépendances (Codespaces):
```bash
pip install -r requirements.txt
