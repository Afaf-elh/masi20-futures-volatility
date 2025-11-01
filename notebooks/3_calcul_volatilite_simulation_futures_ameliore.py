"""
Script de calcul de volatilité et simulation des futures pour plusieurs pays
(Maroc, Vietnam, Turquie, Afrique du Sud, Inde).
Version améliorée avec intégration des modules utils et config.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from arch import arch_model
import warnings
import logging
from typing import Optional, Tuple, List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# Importer les modules utilitaires et de configuration
from utils import (
    charger_donnees, sauvegarder_donnees, tracer_volatilite, 
    tracer_comparaison_volatilites, detecter_valeurs_aberrantes,
    tester_stationnarite, differencier_serie, creer_tableau_bord_interactif
)
from config import (
    CHEMINS, PAYS, MODELES_VOLATILITE, SIMULATION_FUTURES, 
    VALEURS_ABERRANTES, STATIONNARITE, RENDEMENTS, VISUALISATION
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CHEMINS.get('logs', ''), 'volatilite.log'))
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def safe_filename(name: str) -> str:
    """
    Retourne une version sûre d'un composant de nom de fichier en supprimant
    les caractères invalides pour les systèmes de fichiers (Windows notamment)
    et en normalisant les espaces et retours de chariot.
    """
    if not isinstance(name, str):
        name = str(name)
    # Caractères interdits sous Windows: <>:"/\\|?* et les caractères de contrôle
    forbidden = '<>:"/\\|?*'
    cleaned = ''.join(ch for ch in name if ch not in forbidden)
    # Supprimer caractères de contrôle (ord < 32)
    cleaned = ''.join(ch for ch in cleaned if ord(ch) >= 32)
    # Remplacer les séquences d'espaces par un underscore
    cleaned = '_'.join(part for part in cleaned.split())
    # Raccourcir si très long
    if len(cleaned) > 200:
        cleaned = cleaned[:200]
    return cleaned

# ---------------------- Fonctions d'UTILITAIRE ---------------------- #

def tracer_volatilite_individuelle(df: pd.DataFrame, pays: str, col_name: str, method_name: str) -> None:
    """
    Trace et enregistre un graphique pour UNE colonne de volatilité donnée.
    
    Args:
        df: DataFrame contenant la colonne de volatilité
        pays: Nom du pays (pour nommer le fichier)
        col_name: Nom de la colonne dans df (ex: 'volatilite_historique')
        method_name: Nom de la méthode (ex: 'Historique')
    """
    if col_name not in df.columns:
        logger.warning(f"La colonne {col_name} n'existe pas dans le DataFrame.")
        return
    
    # Créer le répertoire de sortie s'il n'existe pas
    pays_dir = PAYS[pays]['dossier']
    output_dir = os.path.join(CHEMINS['volatilite'], pays_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Titre et chemin de sauvegarde
    titre = f"Volatilité {method_name} - {PAYS[pays]['indice']}"
    # Utiliser le suffixe complet après "volatilite_" pour éviter les collisions (ex: 'gjr_garch' vs 'garch')
    suffix = col_name.replace('volatilite_', '').strip('_')
    suffix = suffix.replace(' ', '_')
    chemin_sauvegarde = os.path.join(output_dir, f"volatilite_{suffix}.png")
    
    # Utiliser la fonction utilitaire pour tracer et sauvegarder
    tracer_volatilite(df, col_name, titre, chemin_sauvegarde)
    
    # Créer également une version interactive avec Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[col_name],
        mode='lines',
        name=method_name,
        line=dict(color=PAYS[pays]['couleur'], width=1.5)
    ))
    
    # Ajouter des annotations pour les événements importants si disponibles
    if VISUALISATION['annotations'] and pays in VISUALISATION['evenements']:
        evenements = VISUALISATION['evenements'][pays]
        
        for date_str, description in evenements.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Trouver la valeur de volatilité à cette date ou à la date la plus proche
            if date in df.index:
                valeur = df.loc[date, col_name]
            else:
                # Trouver la date la plus proche
                date_idx = df.index.get_indexer([date], method='nearest')[0]
                date_proche = df.index[date_idx]
                valeur = df.loc[date_proche, col_name]
                date = date_proche
            
            # Ajouter l'annotation
            fig.add_annotation(
                x=date,
                y=valeur,
                text=description,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black',
                ax=0,
                ay=-40
            )
    
    fig.update_layout(
        title=titre,
        xaxis_title='Date',
        yaxis_title='Volatilité',
        template=VISUALISATION['interactivite']['template'],
        height=600,
        width=1000
    )
    
    # Sauvegarder la version interactive
    filename_inter = safe_filename(f"volatilite_{suffix}_interactif.html")
    chemin_interactif = os.path.join(output_dir, filename_inter)
    fig.write_html(chemin_interactif)
    
    logger.info(f"Graphique de volatilité {method_name} pour {PAYS[pays]['nom']} sauvegardé dans {chemin_sauvegarde} et {chemin_interactif}")


def tracer_futures_individuelle(df: pd.DataFrame, pays: str, col_name: str, method_name: str) -> None:
    """
    Trace et enregistre un graphique pour UNE colonne de futures simulés.
    
    Args:
        df: DataFrame contenant la colonne de futures
        pays: Nom du pays (pour nommer le fichier)
        col_name: Nom de la colonne dans df (ex: 'future_cost_of_carry')
        method_name: Nom de la méthode (ex: 'Cost of Carry')
    """
    if col_name not in df.columns or 'close_indice' not in df.columns:
        logger.warning(f"Les colonnes nécessaires n'existent pas pour tracer les futures {method_name}.")
        return
    
    # Créer le répertoire de sortie s'il n'existe pas
    pays_dir = PAYS[pays]['dossier']
    output_dir = os.path.join(CHEMINS['volatilite'], pays_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer la figure
    plt.figure(figsize=(12, 6))
    
    # Tracer l'indice et les futures
    plt.plot(df.index, df['close_indice'], label=f"{PAYS[pays]['indice']} Spot", linewidth=1.5)
    plt.plot(df.index, df[col_name], label=f"{PAYS[pays]['indice']} Future ({method_name})", linewidth=1.5, linestyle='--')
    
    # Ajouter les futures réels si disponibles
    if 'close_future' in df.columns:
        plt.plot(df.index, df['close_future'], label=f"{PAYS[pays]['indice']} Future (Réel)", linewidth=1.5, color='red')
    
    # Ajouter les éléments du graphique
    plt.title(f"Simulation des Futures {method_name} - {PAYS[pays]['indice']}")
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Sauvegarder le graphique
    # Utiliser le suffixe complet après "future_" pour éviter les collisions
    suffix = col_name.replace('future_', '').strip('_')
    suffix = suffix.replace(' ', '_')
    output_file = os.path.join(output_dir, f"future_{suffix}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Créer également une version interactive avec Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close_indice'],
        mode='lines',
        name=f"{PAYS[pays]['indice']} Spot",
        line=dict(color='blue', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[col_name],
        mode='lines',
        name=f"{PAYS[pays]['indice']} Future ({method_name})",
        line=dict(color=PAYS[pays]['couleur'], width=1.5, dash='dash')
    ))
    
    if 'close_future' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close_future'],
            mode='lines',
            name=f"{PAYS[pays]['indice']} Future (Réel)",
            line=dict(color='red', width=1.5)
        ))
    
    # Ajouter des annotations pour les événements importants si disponibles
    if VISUALISATION['annotations'] and pays in VISUALISATION['evenements']:
        evenements = VISUALISATION['evenements'][pays]
        
        for date_str, description in evenements.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Trouver la valeur à cette date ou à la date la plus proche
            if date in df.index:
                valeur = df.loc[date, 'close_indice']
            else:
                # Trouver la date la plus proche
                date_idx = df.index.get_indexer([date], method='nearest')[0]
                date_proche = df.index[date_idx]
                valeur = df.loc[date_proche, 'close_indice']
                date = date_proche
            
            # Ajouter l'annotation
            fig.add_annotation(
                x=date,
                y=valeur,
                text=description,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black',
                ax=0,
                ay=-40
            )
    
    fig.update_layout(
        title=f"Simulation des Futures {method_name} - {PAYS[pays]['indice']}",
        xaxis_title='Date',
        yaxis_title='Prix',
        template=VISUALISATION['interactivite']['template'],
        height=600,
        width=1000
    )
    
    # Sauvegarder la version interactive
    filename_inter = safe_filename(f"future_{suffix}_interactif.html")
    output_file_interactive = os.path.join(output_dir, filename_inter)
    fig.write_html(output_file_interactive)
    
    logger.info(f"Graphique de simulation des futures {method_name} pour {PAYS[pays]['nom']} sauvegardé dans {output_file} et {output_file_interactive}")

def obtenir_colonne_rendements(df: pd.DataFrame) -> Optional[str]:
    """Retourne le nom de la série de rendements à utiliser pour les modèles de volatilité."""
    if 'rendement_stationnaire' in df.columns and df['rendement_stationnaire'].notna().sum() > 0:
        return 'rendement_stationnaire'
    if 'rendement' in df.columns and df['rendement'].notna().sum() > 0:
        logger.warning("Utilisation des rendements bruts faute de série stationnaire explicite.")
        return 'rendement'
    logger.warning("Aucune colonne de rendements disponible pour l'estimation des modèles conditionnels.")
    return None


def verifier_diagnostics_garch(result, nom_modele: str, alpha: float = 0.05) -> None:
    """Effectue des diagnostics de base sur les résidus et les paramètres estimés."""
    try:
        params = result.params
        if params is None:
            logger.warning(f"Diagnostics indisponibles pour {nom_modele}: aucun paramètre estimé.")
            return

        alpha_terms = [params[idx] for idx in params.index if idx.lower().startswith('alpha')]
        beta_terms = [params[idx] for idx in params.index if idx.lower().startswith('beta')]
        if alpha_terms or beta_terms:
            somme_stabilite = sum(alpha_terms) + sum(beta_terms)
            if somme_stabilite >= 1:
                logger.warning(
                    f"Condition de stationnarité violée pour {nom_modele}: somme(alpha)+somme(beta) = {somme_stabilite:.3f}"
                )
            else:
                logger.info(
                    f"Condition alpha+beta<1 respectée pour {nom_modele} (somme = {somme_stabilite:.3f})."
                )

        insignifiants = [name for name, pval in result.pvalues.items() if pval > alpha]
        if insignifiants:
            logger.warning(
                f"Paramètres non significatifs détectés pour {nom_modele} (p-value > {alpha}): {', '.join(insignifiants)}"
            )
        else:
            logger.info(f"Tous les paramètres estimés pour {nom_modele} sont significatifs au seuil {alpha}.")

        resid = result.resid.dropna()
        if len(resid) > 10:
            lb_result = acorr_ljungbox(resid ** 2, lags=[10], return_df=True)
            pvalue = lb_result['lb_pvalue'].iloc[0]
            if pvalue < alpha:
                logger.warning(
                    f"Effets ARCH résiduels détectés pour {nom_modele} (p-value Ljung-Box = {pvalue:.3f})."
                )
            else:
                logger.info(
                    f"Absence d'effets ARCH résiduels détectée pour {nom_modele} (p-value Ljung-Box = {pvalue:.3f})."
                )
        else:
            logger.warning(
                f"Échantillon insuffisant pour appliquer le test de Ljung-Box sur les résidus de {nom_modele}."
            )
    except Exception as exc:
        logger.warning(f"Impossible de réaliser les diagnostics pour {nom_modele}: {exc}")

def charger_donnees_pays(pays: str) -> Optional[pd.DataFrame]:
    """
    Charge les données harmonisées d'un pays.
    
    Args:
        pays: Clé du pays dans le dictionnaire PAYS
        
    Returns:
        DataFrame contenant les données harmonisées ou None en cas d'erreur
    """
    try:
        pays_dir = PAYS[pays]['dossier']
        file_path = os.path.join(CHEMINS['data_harmonisee'], pays_dir, "donnees_fusionnees_final.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Le fichier {file_path} n'existe pas.")
            return None
        
        # Charger les données avec conversion explicite des types
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Nettoyer et convertir les colonnes numériques
        numeric_cols = ['close_indice', 'close_future']
        for col in numeric_cols:
            if col in df.columns:
                # Supprimer les guillemets et les virgules des milliers
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace('"', '', regex=False).str.replace(',', '', regex=False)
                # Convertir en numérique
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Définir la colonne date comme index
        df = df.set_index('date')
        df = df.sort_index()
        
        # Vérifier les valeurs manquantes
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"Valeurs manquantes dans les données de {PAYS[pays]['nom']}:")
            for col, count in missing_values.items():
                if count > 0:
                    logger.info(f"  {col}: {count} valeurs manquantes ({count/len(df)*100:.1f}%)")
        
        logger.info(f"Données chargées pour {PAYS[pays]['nom']}: {df.shape[0]} lignes x {df.shape[1]} colonnes")
        logger.info(f"Période: {df.index.min().strftime('%Y-%m-%d')} à {df.index.max().strftime('%Y-%m-%d')}")
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données pour {PAYS[pays]['nom']}: {e}")
        return None


def calculer_rendements(df: pd.DataFrame, colonne_prix: str = 'close_indice') -> pd.DataFrame:
    """
    Calcule les rendements à partir d'une colonne de prix.
    
    Args:
        df: DataFrame contenant les prix
        colonne_prix: Nom de la colonne contenant les prix
        
    Returns:
        DataFrame avec une colonne 'rendement' ajoutée
    """
    try:
        if colonne_prix not in df.columns:
            logger.warning(f"La colonne {colonne_prix} n'existe pas dans le DataFrame.")
            return df
        
        df_copy = df.copy()
        
        # Calculer les rendements selon la méthode spécifiée dans la configuration
        methode = RENDEMENTS.get('methode', 'log')
        multiplicateur = RENDEMENTS.get('multiplicateur', 100)
        
        if methode == 'log':
            # Rendements logarithmiques
            df_copy['rendement'] = np.log(df_copy[colonne_prix] / df_copy[colonne_prix].shift(1)) * multiplicateur
        else:
            # Rendements simples
            df_copy['rendement'] = df_copy[colonne_prix].pct_change() * multiplicateur

        # Initialiser la colonne stationnaire avec les rendements bruts
        df_copy['rendement_stationnaire'] = df_copy['rendement']
        df_copy['rendement_stationnaire_ordre'] = 0
       
        # Tester la stationnarité des rendements si configuré
        if STATIONNARITE.get('test', 'adf') == 'adf':
            est_stationnaire, p_value, _ = tester_stationnarite(df_copy['rendement'], 'rendements')
            
            if not est_stationnaire:
                logger.warning(
                    f"Les rendements ne sont pas stationnaires (p-value: {p_value:.4f}). Tentative de différenciation jusqu'à {STATIONNARITE.get('max_diff', 0)} ordres."
                )

                max_diff = STATIONNARITE.get('max_diff', 0)
                stationnaire_trouve = False
                for ordre in range(1, max_diff + 1):
                    rendements_diff = differencier_serie(df_copy['rendement'], ordre=ordre)
                    df_copy[f'rendement_diff_{ordre}'] = rendements_diff
                    est_stationnaire_diff, p_value_diff, _ = tester_stationnarite(
                        rendements_diff,
                        f'rendements différenciés (ordre {ordre})'
                    )
                    
                    if est_stationnaire_diff:
                        logger.info(
                            f"Les rendements différenciés d'ordre {ordre} sont stationnaires (p-value: {p_value_diff:.4f})."
                        )
                        df_copy['rendement_stationnaire'] = rendements_diff
                        df_copy['rendement_stationnaire_ordre'] = ordre
                        stationnaire_trouve = True
                        break

                if not stationnaire_trouve:
                    logger.warning(
                        "Aucune différenciation n'a permis d'obtenir une série stationnaire. Utilisation des rendements bruts pour l'estimation."
                    )

        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des rendements: {e}")
        return df


def calculer_volatilite_historique(df: pd.DataFrame, fenetre: int = 30, annualisation: bool = True) -> pd.DataFrame:
    """
    Calcule la volatilité historique avec une fenêtre mobile.
    
    Args:
        df: DataFrame contenant une colonne 'rendement'
        fenetre: Taille de la fenêtre mobile en jours
        annualisation: Si True, annualise la volatilité
        
    Returns:
        DataFrame avec une colonne 'volatilite_historique' ajoutée
    """
    try:
        if 'rendement' not in df.columns:
            logger.warning("La colonne 'rendement' n'existe pas dans le DataFrame.")
            return df
        
        df_copy = df.copy()
        
        # Calculer l'écart-type mobile
        df_copy['volatilite_historique'] = df_copy['rendement'].rolling(window=fenetre).std()
        
        # Annualiser la volatilité si demandé
        if annualisation:
            df_copy['volatilite_historique'] = df_copy['volatilite_historique'] * np.sqrt(252)
        
        # Détecter les valeurs aberrantes si configuré
        if VALEURS_ABERRANTES.get('analyse_separee', False):
            df_copy = detecter_valeurs_aberrantes(
                df_copy, 
                'volatilite_historique', 
                methode=VALEURS_ABERRANTES.get('methode', 'zscore'),
                seuil=VALEURS_ABERRANTES.get('seuil', 3.0),
                traitement=VALEURS_ABERRANTES.get('traitement', 'marquer')
            )
            
            if 'aberrante' in df_copy.columns:
                n_aberrantes = df_copy['aberrante'].sum()
                if n_aberrantes > 0:
                    logger.info(f"Détection de {n_aberrantes} valeurs aberrantes dans la volatilité historique ({n_aberrantes/len(df_copy)*100:.1f}%)")
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la volatilité historique: {e}")
        return df

def optimiser_parametres_garch(
    df: pd.DataFrame,
    p_max: int = 10,
    q_max: int = 10,
    criteres: Tuple[str, ...] = ("aic", "bic")
) -> Tuple[Dict[str, Tuple[int, int]], List[Dict[str, Any]]]:    
    """
        Optimise les paramètres p et q du modèle GARCH en utilisant différents critères d'information.        
        Returns:
            Tuple contenant :
            - un dictionnaire {critère: (p, q)}
            - la liste des résultats détaillés pour chaque couple testé
    """
    try:
        colonne_rendements = obtenir_colonne_rendements(df)
        if colonne_rendements is None:
            return {crit: (1, 1) for crit in criteres}

        rendements = df[colonne_rendements].dropna()

        if len(rendements) < 100:
            logger.warning("Pas assez de données pour optimiser les paramètres GARCH.")
            return {crit: (1, 1) for crit in criteres}, []

        best_scores = {crit: np.inf for crit in criteres}
        best_params = {crit: (1, 1) for crit in criteres}
        resultats = []

        for p in range(0, p_max + 1):
            for q in range(0, q_max + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = arch_model(rendements, vol='Garch', p=p, q=q, rescale=True)
                    result = model.fit(disp='off')
                    
                    scores = {
                        'aic': result.aic,
                        'bic': result.bic
                    }
                    resultats.append({'p': p, 'q': q, **scores})
                    logger.info(f"Scores pour p={p}, q={q}: AIC={scores['aic']:.4f}, BIC={scores['bic']:.4f}")

                    for crit in criteres:
                        valeur = scores.get(crit)
                        if valeur is None:
                            continue
                        if valeur < best_scores[crit]:
                            best_scores[crit] = valeur
                            best_params[crit] = (p, q)
                        
                except Exception as e:
                    logger.warning(f"Erreur lors de l'ajustement du modèle avec p={p}, q={q}: {e}")
        
        for crit in criteres:
            logger.info(f"Meilleurs paramètres selon {crit.upper()} : p={best_params[crit][0]}, q={best_params[crit][1]}")

        return best_params, resultats
    
    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation des paramètres GARCH: {e}")
        return {crit: (1, 1) for crit in criteres}, []


def rechercher_parametres_garch_topk(
    df: pd.DataFrame,
    p_max: int = 10,
    q_max: int = 10,
    top_k: int = 3,
    critere: str = 'aic'
) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    """Retourne le meilleur couple (p, q) et la liste top-k selon le critère choisi."""
    best_params_map, resultats = optimiser_parametres_garch(df, p_max, q_max, (critere, 'bic'))

    if not resultats:
        return best_params_map.get(critere, (1, 1)), [best_params_map.get(critere, (1, 1))]

    critere = critere.lower()
    if critere not in {'aic', 'bic'}:
        critere = 'aic'


    resultats_tries = sorted(resultats, key=lambda item: item[critere])
    top_pairs = [(item['p'], item['q']) for item in resultats_tries[:max(1, top_k)]]
    meilleur = top_pairs[0]
    return meilleur, top_pairs


def calculer_volatilite_garch(
    df: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    optimisation: bool = True,
    p_max: int = 10,
    q_max: int = 10,
    annualisation: bool = True,
    critere: str = 'aic'
) -> pd.DataFrame:
    """
    Calcule la volatilité avec un modèle GARCH.
    
    Args:
        df: DataFrame contenant une colonne 'rendement'
        p: Ordre du terme GARCH
        q: Ordre du terme ARCH
        optimisation: Si True, optimise les paramètres p et q
        p_max: Ordre maximal pour p lors de l'optimisation
        q_max: Ordre maximal pour q lors de l'optimisation
        annualisation: Si True, annualise la volatilité
        
    Returns:
        DataFrame avec une colonne 'volatilite_garch' ajoutée
    """
    try:

        df_copy = df.copy()
        
        colonne_rendements = obtenir_colonne_rendements(df_copy)
        if colonne_rendements is None:
            return df_copy

        rendements = df_copy[colonne_rendements].dropna()
        
        if len(rendements) < 100:  # Pas assez de données
            logger.warning("Pas assez de données pour ajuster un modèle GARCH.")
            return df_copy
        
        # Optimiser les paramètres si demandé
        if optimisation:
            critere = critere.lower()
            best_params_map, _ = optimiser_parametres_garch(df_copy, p_max, q_max, (critere,))
            p, q = best_params_map.get(critere, (p, q))
        
        # Ajuster le modèle GARCH
        model = arch_model(rendements, vol='Garch', p=p, q=q, rescale=True)
        result = model.fit(disp='off')
        verifier_diagnostics_garch(result, f"GARCH(p={p}, q={q})")

        # Extraire la volatilité conditionnelle
        volatilite = result.conditional_volatility
        
        # Créer une série avec le même index que le DataFrame original
        volatilite_series = volatilite.reindex(df_copy.index)
        volatilite_series = volatilite_series.interpolate(method='linear')
        
        # Ajouter la volatilité au DataFrame
        df_copy['volatilite_garch'] = volatilite_series
        df_copy['garch_p'] = p
        df_copy['garch_q'] = q
        df_copy['garch_rendements_source'] = colonne_rendements

        # Annualiser la volatilité si demandé
        if annualisation:
            df_copy['volatilite_garch'] = df_copy['volatilite_garch'] * np.sqrt(252)
        
        # Détecter les valeurs aberrantes si configuré
        if VALEURS_ABERRANTES.get('analyse_separee', False):
            df_copy = detecter_valeurs_aberrantes(
                df_copy, 
                'volatilite_garch', 
                methode=VALEURS_ABERRANTES.get('methode', 'zscore'),
                seuil=VALEURS_ABERRANTES.get('seuil', 3.0),
                traitement=VALEURS_ABERRANTES.get('traitement', 'marquer')
            )
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la volatilité GARCH: {e}")
        return df


def calculer_volatilite_egarch(
    df: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    optimisation: bool = True,
    annualisation: bool = True,
    p_max: int = 5,
    q_max: int = 5,
    candidates: Optional[List[Tuple[int, int]]] = None,
    critere: str = 'aic'
) -> pd.DataFrame:
    """
    Calcule la volatilité avec un modèle EGARCH en testant plusieurs couples (p, q).
    """
    try:
        
        df_copy = df.copy()
        
        colonne_rendements = obtenir_colonne_rendements(df_copy)
        if colonne_rendements is None:
            return df_copy

        rendements = df_copy[colonne_rendements].dropna()

        if len(rendements) < 100:
            logger.warning("Pas assez de données pour ajuster un modèle EGARCH.")
            return df_copy
        
        recherche = []
        if candidates:
            recherche = list(dict.fromkeys(candidates))

        if optimisation:
            critere = critere.lower()
            if not recherche:
                for p_cand in range(0, p_max + 1):
                    for q_cand in range(0, q_max + 1):
                        if p_cand == 0 and q_cand == 0:
                            continue
                        recherche.append((p_cand, q_cand))
        elif not recherche:
            recherche = [(p, q)]

        meilleur_result = None
        meilleur_score = np.inf
        meilleur_params = (p, q)

        for cand_p, cand_q in recherche:
            try:
                model = arch_model(rendements, vol='EGARCH', p=cand_p, q=cand_q, rescale=True)
                result = model.fit(disp='off')
                score = getattr(result, critere.lower(), np.inf)
                logger.info(f"EGARCH scores p={cand_p}, q={cand_q}: AIC={result.aic:.4f}, BIC={result.bic:.4f}")
                if score < meilleur_score:
                    meilleur_score = score
                    meilleur_result = result
                    meilleur_params = (cand_p, cand_q)
            except Exception as exc:
                logger.warning(f"Erreur lors de l'ajustement EGARCH p={cand_p}, q={cand_q}: {exc}")

        if meilleur_result is None:
            logger.warning("Aucun ajustement EGARCH concluant n'a été trouvé.")
            return df_copy

        verifier_diagnostics_garch(meilleur_result, f"EGARCH(p={meilleur_params[0]}, q={meilleur_params[1]})")

        volatilite = meilleur_result.conditional_volatility
        volatilite_series = volatilite.reindex(df_copy.index).interpolate(method='linear')
        df_copy['volatilite_egarch'] = volatilite_series
        df_copy['egarch_p'] = meilleur_params[0]
        df_copy['egarch_q'] = meilleur_params[1]
        df_copy['egarch_rendements_source'] = colonne_rendements

        if annualisation:
            df_copy['volatilite_egarch'] = df_copy['volatilite_egarch'] * np.sqrt(252)

        if VALEURS_ABERRANTES.get('analyse_separee', False):
            df_copy = detecter_valeurs_aberrantes(
                df_copy,
                'volatilite_egarch',
                methode=VALEURS_ABERRANTES.get('methode', 'zscore'),
                seuil=VALEURS_ABERRANTES.get('seuil', 3.0),
                traitement=VALEURS_ABERRANTES.get('traitement', 'marquer')
            )

        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la volatilité EGARCH: {e}")
        return df


def calculer_volatilite_gjr_garch(
    df: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    optimisation: bool = True,
    annualisation: bool = True,
    p_max: int = 5,
    q_max: int = 5,
    candidates: Optional[List[Tuple[int, int]]] = None,
    critere: str = 'aic'
) -> pd.DataFrame:
    """
    Calcule la volatilité avec un modèle GJR-GARCH en évaluant plusieurs couples (p, q).
    """
    try:
        
        df_copy = df.copy()
        
        colonne_rendements = obtenir_colonne_rendements(df_copy)
        if colonne_rendements is None:
            return df_copy

        rendements = df_copy[colonne_rendements].dropna()

        if len(rendements) < 100:
            logger.warning("Pas assez de données pour ajuster un modèle GJR-GARCH.")
            return df_copy
        
        recherche = []
        if candidates:
            recherche = list(dict.fromkeys(candidates))

        if optimisation:
            critere = critere.lower()
            if not recherche:
                for p_cand in range(0, p_max + 1):
                    for q_cand in range(0, q_max + 1):
                        if p_cand == 0 and q_cand == 0:
                            continue
                        recherche.append((p_cand, q_cand))
        elif not recherche:
            recherche = [(p, q)]

        meilleur_result = None
        meilleur_score = np.inf
        meilleur_params = (p, q)

        for cand_p, cand_q in recherche:
            try:
                model = arch_model(rendements, vol='GARCH', p=cand_p, q=cand_q, o=1, power=2.0, rescale=True)
                result = model.fit(disp='off')
                score = getattr(result, critere.lower(), np.inf)
                logger.info(f"GJR-GARCH scores p={cand_p}, q={cand_q}: AIC={result.aic:.4f}, BIC={result.bic:.4f}")
                if score < meilleur_score:
                    meilleur_score = score
                    meilleur_result = result
                    meilleur_params = (cand_p, cand_q)
            except Exception as exc:
                logger.warning(f"Erreur lors de l'ajustement GJR-GARCH p={cand_p}, q={cand_q}: {exc}")

        if meilleur_result is None:
            logger.warning("Aucun ajustement GJR-GARCH concluant n'a été trouvé.")
            return df_copy

        verifier_diagnostics_garch(meilleur_result, f"GJR-GARCH(p={meilleur_params[0]}, q={meilleur_params[1]})")

        volatilite = meilleur_result.conditional_volatility
        volatilite_series = volatilite.reindex(df_copy.index).interpolate(method='linear')
        df_copy['volatilite_gjr_garch'] = volatilite_series
        df_copy['gjr_garch_p'] = meilleur_params[0]
        df_copy['gjr_garch_q'] = meilleur_params[1]
        df_copy['gjr_rendements_source'] = colonne_rendements

        if annualisation:
            df_copy['volatilite_gjr_garch'] = df_copy['volatilite_gjr_garch'] * np.sqrt(252)

        if VALEURS_ABERRANTES.get('analyse_separee', False):
            df_copy = detecter_valeurs_aberrantes(
                df_copy,
                'volatilite_gjr_garch',
                methode=VALEURS_ABERRANTES.get('methode', 'zscore'),
                seuil=VALEURS_ABERRANTES.get('seuil', 3.0),
                traitement=VALEURS_ABERRANTES.get('traitement', 'marquer')
            )

        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la volatilité GJR-GARCH: {e}")
        return df


def simuler_futures_cost_of_carry(df: pd.DataFrame, pays: str, maturite: int = 90) -> pd.DataFrame:
    """
    Simule les prix des futures avec le modèle Cost of Carry.
    
    Args:
        df: DataFrame contenant les données
        pays: Clé du pays dans le dictionnaire PAYS
        maturite: Maturité des futures en jours
        
    Returns:
        DataFrame avec une colonne 'future_cost_of_carry' ajoutée
    """
    try:
        if 'close_indice' not in df.columns:
            logger.warning("La colonne 'close_indice' n'existe pas dans le DataFrame.")
            return df
        
        df_copy = df.copy()
        
        # Récupérer le taux sans risque pour le pays
        taux_sans_risque = SIMULATION_FUTURES['cost_of_carry']['taux_sans_risque'].get(pays, 0.03)
        
        # Calculer le prix des futures
        df_copy['future_cost_of_carry'] = df_copy['close_indice'] * np.exp(taux_sans_risque * maturite / 365)
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors de la simulation des futures avec le modèle Cost of Carry: {e}")
        return df


def simuler_futures_convenience_yield(df: pd.DataFrame, pays: str, maturite: int = 90) -> pd.DataFrame:
    """
    Simule les prix des futures avec le modèle Convenience Yield.
    
    Args:
        df: DataFrame contenant les données
        pays: Clé du pays dans le dictionnaire PAYS
        maturite: Maturité des futures en jours
        
    Returns:
        DataFrame avec une colonne 'future_convenience_yield' ajoutée
    """
    try:
        if 'close_indice' not in df.columns:
            logger.warning("La colonne 'close_indice' n'existe pas dans le DataFrame.")
            return df
        
        df_copy = df.copy()
        
        # Récupérer les paramètres pour le pays
        taux_sans_risque = SIMULATION_FUTURES['convenience_yield']['taux_sans_risque'].get(pays, 0.03)
        storage_cost = SIMULATION_FUTURES['convenience_yield']['storage_cost']
        convenience_yield = SIMULATION_FUTURES['convenience_yield']['convenience_yield'].get(pays, 0.02)
        
        # Calculer le prix des futures
        df_copy['future_convenience_yield'] = df_copy['close_indice'] * np.exp((taux_sans_risque + storage_cost - convenience_yield) * maturite / 365)
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors de la simulation des futures avec le modèle Convenience Yield: {e}")
        return df


def simuler_futures_monte_carlo(df: pd.DataFrame, pays: str, maturite: int = 90, n_simulations: int = 1000) -> pd.DataFrame:
    """
    Simule les prix des futures avec une simulation Monte Carlo.
    
    Args:
        df: DataFrame contenant les données
        pays: Clé du pays dans le dictionnaire PAYS
        maturite: Maturité des futures en jours
        n_simulations: Nombre de simulations
        
    Returns:
        DataFrame avec une colonne 'future_monte_carlo' ajoutée
    """
    try:
        if 'close_indice' not in df.columns or 'volatilite_historique' not in df.columns:
            logger.warning("Les colonnes nécessaires n'existent pas dans le DataFrame.")
            return df
        
        df_copy = df.copy()
        
        # Récupérer les paramètres pour le pays
        taux_sans_risque = SIMULATION_FUTURES['monte_carlo']['taux_sans_risque'].get(pays, 0.03)
        vol_of_vol = SIMULATION_FUTURES['monte_carlo']['vol_of_vol']
        mean_reversion = SIMULATION_FUTURES['monte_carlo']['mean_reversion']
        
        # Initialiser les résultats
        futures_prices = np.zeros(len(df_copy))
        
        # Pour chaque date
        for i in range(len(df_copy)):
            # Récupérer le prix de l'indice et la volatilité
            spot_price = df_copy['close_indice'].iloc[i]
            volatility = df_copy['volatilite_historique'].iloc[i]
            
            if pd.isna(spot_price) or pd.isna(volatility):
                futures_prices[i] = np.nan
                continue
            
            # Simuler les trajectoires
            dt = 1/252  # Pas de temps journalier
            n_steps = int(maturite / 365 * 252)  # Nombre de pas de temps
            
            if n_steps <= 0:
                futures_prices[i] = spot_price
                continue
            
            # Initialiser les simulations
            simulated_prices = np.zeros((n_simulations, n_steps + 1))
            simulated_prices[:, 0] = spot_price
            
            # Simuler les trajectoires
            for t in range(1, n_steps + 1):
                # Simuler la volatilité stochastique
                vol_shock = np.random.normal(0, vol_of_vol, n_simulations)
                volatility = np.maximum(0.001, volatility + mean_reversion * (0.2 - volatility) * dt + vol_shock * np.sqrt(dt))
                
                # Simuler les prix
                price_shock = np.random.normal(0, volatility, n_simulations)
                simulated_prices[:, t] = simulated_prices[:, t-1] * np.exp((taux_sans_risque - 0.5 * volatility**2) * dt + price_shock * np.sqrt(dt))
            
            # Calculer le prix du future comme la moyenne des prix simulés à maturité
            futures_prices[i] = np.mean(simulated_prices[:, -1])
        
        # Ajouter les prix des futures au DataFrame
        df_copy['future_monte_carlo'] = futures_prices
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors de la simulation des futures avec Monte Carlo: {e}")
        return df


def comparer_volatilites(df: pd.DataFrame, pays: str) -> None:
    """
    Compare les différentes méthodes de calcul de volatilité.
    
    Args:
        df: DataFrame contenant les colonnes de volatilité
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Identifier les colonnes de volatilité
        volatility_cols = [col for col in df.columns if col.startswith('volatilite_')]
        
        if not volatility_cols:
            logger.warning("Aucune colonne de volatilité trouvée dans le DataFrame.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        pays_dir = PAYS[pays]['dossier']
        output_dir = os.path.join(CHEMINS['volatilite'], pays_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Utiliser la fonction utilitaire pour tracer et sauvegarder
        titre = f"Comparaison des volatilités - {PAYS[pays]['indice']}"
        chemin_sauvegarde = os.path.join(output_dir, "comparaison_volatilites.png")
        
        tracer_comparaison_volatilites(df, volatility_cols, titre, chemin_sauvegarde)
        
        # Créer également une version interactive avec Plotly
        fig = go.Figure()
        
        for col in volatility_cols:
            nom_methode = col.replace('volatilite_', '').replace('_', ' ').title()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=nom_methode
            ))
        
        # Ajouter des annotations pour les événements importants si disponibles
        if VISUALISATION['annotations'] and pays in VISUALISATION['evenements']:
            evenements = VISUALISATION['evenements'][pays]
            
            for date_str, description in evenements.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                
                # Trouver la valeur de volatilité à cette date ou à la date la plus proche
                if date in df.index:
                    valeur = df.loc[date, volatility_cols[0]]
                else:
                    # Trouver la date la plus proche
                    date_idx = df.index.get_indexer([date], method='nearest')[0]
                    date_proche = df.index[date_idx]
                    valeur = df.loc[date_proche, volatility_cols[0]]
                    date = date_proche
                
                # Ajouter l'annotation
                fig.add_annotation(
                    x=date,
                    y=valeur,
                    text=description,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='black',
                    ax=0,
                    ay=-40
                )
        
        fig.update_layout(
            title=titre,
            xaxis_title='Date',
            yaxis_title='Volatilité',
            template=VISUALISATION['interactivite']['template'],
            height=600,
            width=1000
        )

        # Sauvegarder la version interactive
        chemin_interactif = os.path.join(output_dir, safe_filename("comparaison_volatilites.html"))
        fig.write_html(chemin_interactif)

        logger.info(f"Graphique de comparaison des volatilités pour {PAYS[pays]['nom']} sauvegardé dans {chemin_sauvegarde} et {chemin_interactif}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la comparaison des volatilités: {e}")


def comparer_futures(df: pd.DataFrame, pays: str) -> None:
    """
    Compare les différentes méthodes de simulation des futures.
    
    Args:
        df: DataFrame contenant les colonnes de futures
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Identifier les colonnes de futures
        futures_cols = [col for col in df.columns if col.startswith('future_')]
        
        if not futures_cols:
            logger.warning("Aucune colonne de futures trouvée dans le DataFrame.")
            return
        
        if 'close_indice' not in df.columns:
            logger.warning("La colonne 'close_indice' n'existe pas dans le DataFrame.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        pays_dir = PAYS[pays]['dossier']
        output_dir = os.path.join(CHEMINS['volatilite'], pays_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Créer la figure
        plt.figure(figsize=(12, 6))
        
        # Tracer l'indice
        plt.plot(df.index, df['close_indice'], label=f"{PAYS[pays]['indice']} Spot", linewidth=1.5)
        
        # Tracer les futures simulés
        for col in futures_cols:
            nom_methode = col.replace('future_', '').replace('_', ' ').title()
            plt.plot(df.index, df[col], label=f"{PAYS[pays]['indice']} Future ({nom_methode})", linewidth=1.5, linestyle='--')
        
        # Tracer les futures réels si disponibles
        if 'close_future' in df.columns:
            plt.plot(df.index, df['close_future'], label=f"{PAYS[pays]['indice']} Future (Réel)", linewidth=1.5, color='red')
        
        # Ajouter les éléments du graphique
        plt.title(f"Comparaison des Futures - {PAYS[pays]['indice']}")
        plt.xlabel('Date')
        plt.ylabel('Prix')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_file = os.path.join(output_dir, "comparaison_futures.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Créer également une version interactive avec Plotly
        fig = go.Figure()
        
        # Tracer l'indice
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close_indice'],
            mode='lines',
            name=f"{PAYS[pays]['indice']} Spot",
            line=dict(color='blue', width=1.5)
        ))
        
        # Tracer les futures simulés
        for col in futures_cols:
            nom_methode = col.replace('future_', '').replace('_', ' ').title()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=f"{PAYS[pays]['indice']} Future ({nom_methode})",
                line=dict(dash='dash', width=1.5)
            ))
        
        # Tracer les futures réels si disponibles
        if 'close_future' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['close_future'],
                mode='lines',
                name=f"{PAYS[pays]['indice']} Future (Réel)",
                line=dict(color='red', width=1.5)
            ))
        
        # Ajouter des annotations pour les événements importants si disponibles
        if VISUALISATION['annotations'] and pays in VISUALISATION['evenements']:
            evenements = VISUALISATION['evenements'][pays]
            
            for date_str, description in evenements.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                
                # Trouver la valeur à cette date ou à la date la plus proche
                if date in df.index:
                    valeur = df.loc[date, 'close_indice']
                else:
                    # Trouver la date la plus proche
                    date_idx = df.index.get_indexer([date], method='nearest')[0]
                    date_proche = df.index[date_idx]
                    valeur = df.loc[date_proche, 'close_indice']
                    date = date_proche
                
                # Ajouter l'annotation
                fig.add_annotation(
                    x=date,
                    y=valeur,
                    text=description,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='black',
                    ax=0,
                    ay=-40
                )
        
        fig.update_layout(
            title=f"Comparaison des Futures - {PAYS[pays]['indice']}",
            xaxis_title='Date',
            yaxis_title='Prix',
            template=VISUALISATION['interactivite']['template'],
            height=600,
            width=1000
        )

        # Sauvegarder la version interactive
        output_file_interactive = os.path.join(output_dir, safe_filename("comparaison_futures.html"))
        fig.write_html(output_file_interactive)

        logger.info(f"Graphique de comparaison des futures pour {PAYS[pays]['nom']} sauvegardé dans {output_file} et {output_file_interactive}")

    except Exception as e:
        logger.error(f"Erreur lors de la comparaison des futures: {e}")


def creer_tableau_bord_pays(df: pd.DataFrame, pays: str) -> None:
    """
    Crée un tableau de bord interactif pour un pays.
    
    Args:
        df: DataFrame contenant les données
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Vérifier que les colonnes nécessaires existent
        volatility_cols = [col for col in df.columns if col.startswith('volatilite_')]
        
        if not volatility_cols:
            logger.warning("Aucune colonne de volatilité trouvée dans le DataFrame.")
            return
        
        if 'close_indice' not in df.columns:
            logger.warning("La colonne 'close_indice' n'existe pas dans le DataFrame.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        pays_dir = PAYS[pays]['dossier']
        output_dir = os.path.join(CHEMINS['volatilite'], pays_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Chemin de sauvegarde
        chemin_sauvegarde = os.path.join(output_dir, f"dashboard_{pays.lower()}.html")
        
        # Créer le tableau de bord interactif
        creer_tableau_bord_interactif(
            df=df,
            titre=f"Tableau de bord - {PAYS[pays]['nom']}",
            colonnes_volatilite=volatility_cols,
            colonne_prix='close_indice',
            chemin_sauvegarde=chemin_sauvegarde
        )
        
        logger.info(f"Tableau de bord interactif pour {PAYS[pays]['nom']} sauvegardé dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la création du tableau de bord: {e}")


def sauvegarder_resultats_volatilite(df: pd.DataFrame, pays: str) -> None:
    """
    Sauvegarde les résultats de volatilité dans un fichier CSV.
    
    Args:
        df: DataFrame contenant les résultats
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Créer le répertoire de sortie s'il n'existe pas
        pays_dir = PAYS[pays]['dossier']
        output_dir = os.path.join(CHEMINS['volatilite'], pays_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Chemin de sauvegarde
        output_file = os.path.join(output_dir, "resultats_volatilite.csv")
        
        # Sélectionner les colonnes à sauvegarder
        cols_to_save = ['close_indice', 'close_future', 'rendement']
        cols_to_save.extend([col for col in df.columns if col.startswith('volatilite_')])
        cols_to_save.extend([col for col in df.columns if col.startswith('future_')])
        
        # Filtrer les colonnes qui existent
        cols_to_save = [col for col in cols_to_save if col in df.columns]
        
        # Réinitialiser l'index pour avoir la date comme colonne
        df_to_save = df[cols_to_save].reset_index()
        
        # Sauvegarder les résultats
        df_to_save.to_csv(output_file, index=False)
        
        logger.info(f"Résultats de volatilité pour {PAYS[pays]['nom']} sauvegardés dans {output_file}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des résultats de volatilité: {e}")


def comparer_volatilites_pays(pays_list: List[str]) -> None:
    """
    Compare les volatilités entre différents pays.
    
    Args:
        pays_list: Liste des clés des pays à comparer
    """
    try:
        # Créer un DataFrame pour stocker les volatilités
        volatilites = pd.DataFrame()
        
        # Charger les volatilités pour chaque pays
        for pays in pays_list:
            pays_dir = PAYS[pays]['dossier']
            file_path = os.path.join(CHEMINS['volatilite'], pays_dir, "resultats_volatilite.csv")
            
            if not os.path.exists(file_path):
                logger.warning(f"Le fichier {file_path} n'existe pas.")
                continue
            
            # Charger les données
            df = pd.read_csv(file_path)
            
            # Vérifier que les colonnes nécessaires existent
            if 'date' not in df.columns or 'volatilite_historique' not in df.columns:
                logger.warning(f"Les colonnes nécessaires n'existent pas dans le fichier {file_path}.")
                continue
            
            # Convertir la date en datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Ajouter la volatilité au DataFrame global
            if volatilites.empty:
                volatilites = pd.DataFrame(index=df['date'])
            
            volatilites[PAYS[pays]['nom']] = df.set_index('date')['volatilite_historique']
        
        if volatilites.empty:
            logger.warning("Aucune donnée de volatilité n'a pu être chargée.")
            return
        
        # Trier l'index par ordre croissant
        volatilites = volatilites.sort_index()
        
        # Interpoler les valeurs manquantes
        volatilites = volatilites.interpolate(method='time')
        
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = CHEMINS['volatilite']
        os.makedirs(output_dir, exist_ok=True)
        
        # Tracer la comparaison des volatilités
        plt.figure(figsize=(12, 6))
        
        for pays in volatilites.columns:
            plt.plot(volatilites.index, volatilites[pays], label=pays, linewidth=1.5)
        
        plt.title("Comparaison des volatilités entre pays")
        plt.xlabel('Date')
        plt.ylabel('Volatilité')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_file = os.path.join(output_dir, "comparaison_volatilites_pays.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Créer également une version interactive avec Plotly
        fig = go.Figure()
        
        for pays in volatilites.columns:
            pays_key = next((k for k, v in PAYS.items() if v['nom'] == pays), None)
            couleur = PAYS[pays_key]['couleur'] if pays_key else None
            
            fig.add_trace(go.Scatter(
                x=volatilites.index,
                y=volatilites[pays],
                mode='lines',
                name=pays,
                line=dict(color=couleur, width=1.5) if couleur else None
            ))
        
        fig.update_layout(
            title="Comparaison des volatilités entre pays",
            xaxis_title='Date',
            yaxis_title='Volatilité',
            template=VISUALISATION['interactivite']['template'],
            height=600,
            width=1000
        )

        # Sauvegarder la version interactive
        output_file_interactive = os.path.join(output_dir, safe_filename("comparaison_volatilites_pays.html"))
        fig.write_html(output_file_interactive)

        # Calculer la matrice de corrélation
        corr = volatilites.corr()
        
        # Tracer la matrice de corrélation
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                   fmt='.2f', square=True, linewidths=0.5, cbar_kws={'label': 'Coefficient de corrélation'})
        plt.title("Corrélation des volatilités entre pays")
        plt.tight_layout()
        
        # Sauvegarder la matrice de corrélation
        output_file_corr = os.path.join(output_dir, "correlation_volatilites_pays.png")
        plt.savefig(output_file_corr, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Créer également une version interactive avec Plotly
        fig_corr = px.imshow(
            corr,
            x=corr.columns,
            y=corr.index,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            text_auto='.2f'
        )
        
        fig_corr.update_layout(
            title="Corrélation des volatilités entre pays",
            xaxis_title="Pays",
            yaxis_title="Pays",
            template=VISUALISATION['interactivite']['template'],
            width=800,
            height=800
        )
        
        # Sauvegarder la version interactive
        output_file_corr_interactive = os.path.join(output_dir, safe_filename("correlation_volatilites_pays.html"))
        fig_corr.write_html(output_file_corr_interactive)

        logger.info(f"Comparaison des volatilités entre pays sauvegardée dans {output_file} et {output_file_interactive}")
        logger.info(f"Matrice de corrélation des volatilités entre pays sauvegardée dans {output_file_corr} et {output_file_corr_interactive}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la comparaison des volatilités entre pays: {e}")


def main():
    """
    Fonction principale.
    """
    try:
        # Liste des pays à traiter
        pays_list = list(PAYS.keys())
        
        # Pour chaque pays
        for pays in pays_list:
            logger.info(f"Traitement du pays: {pays}")
            
            # Charger les données
            df = charger_donnees_pays(pays)
            
            if df is None:
                logger.warning(f"Impossible de charger les données pour {pays}. Passage au pays suivant.")
                continue
            
            # Calculer les rendements
            df = calculer_rendements(df)
            
            # Calculer les volatilités
            # Volatilité historique
            params_historique = MODELES_VOLATILITE['historique']
            df = calculer_volatilite_historique(
                df, 
                fenetre=params_historique['fenetre'], 
                annualisation=params_historique['annualisation']
            )
            tracer_volatilite_individuelle(df, pays, 'volatilite_historique', 'Historique')
            
          
            # Volatilité GARCH avec optimisation séparée AIC/BIC
            params_garch = MODELES_VOLATILITE['garch']
            critere_garch = params_garch.get('critere', 'aic')
            p_max_garch = params_garch.get('p_max', 10)
            q_max_garch = params_garch.get('q_max', 10)
            top_k_pairs = [(params_garch.get('p', 1), params_garch.get('q', 1))]

            if params_garch.get('optimisation', False):
                meilleur, top_k_pairs = rechercher_parametres_garch_topk(
                    df,
                    p_max=p_max_garch,
                    q_max=q_max_garch,
                    top_k=params_garch.get('top_k', 3),
                    critere=critere_garch
                )
                logger.info(f"Meilleurs paramètres GARCH selon {critere_garch.upper()} pour {pays}: p={meilleur[0]}, q={meilleur[1]}")
            else:
                meilleur = (params_garch.get('p', 1), params_garch.get('q', 1))

            top_k_pairs = list(dict.fromkeys(top_k_pairs))

            df = calculer_volatilite_garch(
                df,
                p=meilleur[0],
                q=meilleur[1],
                optimisation=False,
                p_max=p_max_garch,
                q_max=q_max_garch,
                annualisation=params_garch.get('annualisation', True),
                critere=critere_garch
            )
            tracer_volatilite_individuelle(df, pays, 'volatilite_garch', 'GARCH')
            
            # Volatilité EGARCH
            params_egarch = MODELES_VOLATILITE['egarch']
            df = calculer_volatilite_egarch(
                df,
                p=params_egarch.get('p', 1),
                q=params_egarch.get('q', 1),
                optimisation=params_egarch.get('optimisation', False),
                annualisation=params_egarch.get('annualisation', True),
                p_max=params_egarch.get('p_max', p_max_garch),
                q_max=params_egarch.get('q_max', q_max_garch),
                candidates=top_k_pairs,
                critere=params_egarch.get('critere', critere_garch)
            )
            tracer_volatilite_individuelle(df, pays, 'volatilite_egarch', 'EGARCH')
            

            # Volatilité GJR-GARCH
            params_gjr_garch = MODELES_VOLATILITE['gjr_garch']
            df = calculer_volatilite_gjr_garch(
                df,
                p=params_gjr_garch.get('p', 1),
                q=params_gjr_garch.get('q', 1),
                optimisation=params_gjr_garch.get('optimisation', False),
                annualisation=params_gjr_garch.get('annualisation', True),
                p_max=params_gjr_garch.get('p_max', p_max_garch),
                q_max=params_gjr_garch.get('q_max', q_max_garch),
                candidates=top_k_pairs,
                critere=params_gjr_garch.get('critere', critere_garch)
            )
            tracer_volatilite_individuelle(df, pays, 'volatilite_gjr_garch', 'GJR-GARCH')

            # Comparer les volatilités
            comparer_volatilites(df, pays)
            
            # Simuler les futures
            # Cost of Carry
            params_cost_of_carry = SIMULATION_FUTURES['cost_of_carry']
            df = simuler_futures_cost_of_carry(
                df, 
                pays, 
                maturite=params_cost_of_carry['maturite']
            )
            tracer_futures_individuelle(df, pays, 'future_cost_of_carry', 'Cost of Carry')
            
            # Convenience Yield
            params_convenience_yield = SIMULATION_FUTURES['convenience_yield']
            df = simuler_futures_convenience_yield(
                df, 
                pays, 
                maturite=params_convenience_yield['maturite']
            )
            tracer_futures_individuelle(df, pays, 'future_convenience_yield', 'Convenience Yield')
            
            # Monte Carlo
            params_monte_carlo = SIMULATION_FUTURES['monte_carlo']
            df = simuler_futures_monte_carlo(
                df, 
                pays, 
                maturite=params_monte_carlo['maturite'],
                n_simulations=params_monte_carlo['n_simulations']
            )
            tracer_futures_individuelle(df, pays, 'future_monte_carlo', 'Monte Carlo')
            
            # Comparer les futures
            comparer_futures(df, pays)
            
            # Créer un tableau de bord interactif
            creer_tableau_bord_pays(df, pays)
            
            # Sauvegarder les résultats
            sauvegarder_resultats_volatilite(df, pays)
            
            logger.info(f"Traitement terminé pour {pays}")
        
        # Comparer les volatilités entre pays
        comparer_volatilites_pays(pays_list)
        
        logger.info("Traitement terminé pour tous les pays")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")


if __name__ == "__main__":
    main()
