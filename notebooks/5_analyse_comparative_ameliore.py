"""
Script d'analyse comparative des volatilités entre différents pays.
Version améliorée avec intégration des modules utils et config.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import warnings
import logging
from typing import Optional, Tuple, List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Importer les modules utilitaires et de configuration
from utils import (
    charger_donnees, sauvegarder_donnees, tracer_heatmap, 
    tester_stationnarite, differencier_serie, generer_statistiques_descriptives
)
from config import (
    CHEMINS, PAYS, ANALYSE_COMPARATIVE, STATIONNARITE, 
    VALEURS_ABERRANTES, VISUALISATION
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CHEMINS.get('logs', ''), 'analyse_comparative.log'))
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def charger_donnees_volatilite_pays() -> Optional[pd.DataFrame]:
    """
    Charge les données de volatilité pour tous les pays.
    
    Returns:
        DataFrame contenant les volatilités de tous les pays ou None en cas d'erreur
    """
    try:
        # Créer un DataFrame pour stocker les volatilités
        volatilites = pd.DataFrame()
        
        # Pour chaque pays
        for pays in PAYS.keys():
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
            return None
        
        # Trier l'index par ordre croissant
        volatilites = volatilites.sort_index()
        
        # Interpoler les valeurs manquantes
        volatilites = volatilites.interpolate(method='time')
        
        logger.info(f"Données de volatilité chargées pour {len(volatilites.columns)} pays: {', '.join(volatilites.columns)}")
        logger.info(f"Période: {volatilites.index.min().strftime('%Y-%m-%d')} à {volatilites.index.max().strftime('%Y-%m-%d')}")
        
        return volatilites
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données de volatilité: {e}")
        return None

def calculer_correlation_dynamique(df: pd.DataFrame, fenetre: int = 60) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Calcule la corrélation dynamique entre les volatilités des différents pays.
    
    Args:
        df: DataFrame contenant les volatilités des pays
        fenetre: Taille de la fenêtre mobile en jours
        
    Returns:
        Dictionnaire contenant les DataFrames de corrélation dynamique pour chaque paire de pays
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df is None or df.empty:
            logger.warning("Le DataFrame est vide ou None.")
            return None
        
        # Vérifier qu'il y a au moins deux pays
        if len(df.columns) < 2:
            logger.warning("Il faut au moins deux pays pour calculer la corrélation dynamique.")
            return None
        
        # Créer un dictionnaire pour stocker les corrélations dynamiques
        correlations = {}
        
        # Pour chaque paire de pays
        for i, pays1 in enumerate(df.columns):
            for j, pays2 in enumerate(df.columns):
                if i < j:  # Éviter les doublons et les auto-corrélations
                    # Calculer la corrélation mobile
                    corr = df[pays1].rolling(window=fenetre).corr(df[pays2])
                    
                    # Stocker la corrélation
                    correlations[f"{pays1}-{pays2}"] = corr
        
        logger.info(f"Corrélation dynamique calculée pour {len(correlations)} paires de pays avec une fenêtre de {fenetre} jours")
        
        return correlations
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la corrélation dynamique: {e}")
        return None

def analyser_contagion(df: pd.DataFrame, seuil_percentile: int = 95) -> Optional[pd.DataFrame]:
    """
    Analyse la contagion entre marchés en période de forte volatilité.
    
    Args:
        df: DataFrame contenant les volatilités des pays
        seuil_percentile: Percentile pour définir les périodes de forte volatilité
        
    Returns:
        DataFrame contenant les probabilités conditionnelles de contagion
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df is None or df.empty:
            logger.warning("Le DataFrame est vide ou None.")
            return None
        
        # Vérifier qu'il y a au moins deux pays
        if len(df.columns) < 2:
            logger.warning("Il faut au moins deux pays pour analyser la contagion.")
            return None
        
        # Calculer les seuils de forte volatilité pour chaque pays
        seuils = {}
        for pays in df.columns:
            seuils[pays] = np.percentile(df[pays].dropna(), seuil_percentile)
        
        # Créer un DataFrame pour les périodes de forte volatilité
        forte_volatilite = pd.DataFrame(index=df.index)
        
        for pays in df.columns:
            forte_volatilite[pays] = df[pays] > seuils[pays]
        
        # Calculer les probabilités conditionnelles
        proba_conditionnelles = pd.DataFrame(index=df.columns, columns=df.columns)
        
        for pays1 in df.columns:
            for pays2 in df.columns:
                if pays1 != pays2:
                    # P(pays2 en forte volatilité | pays1 en forte volatilité)
                    n_pays1 = forte_volatilite[pays1].sum()
                    n_pays1_et_pays2 = (forte_volatilite[pays1] & forte_volatilite[pays2]).sum()
                    
                    if n_pays1 > 0:
                        proba_conditionnelles.loc[pays1, pays2] = n_pays1_et_pays2 / n_pays1
                    else:
                        proba_conditionnelles.loc[pays1, pays2] = np.nan
        
        proba_conditionnelles = proba_conditionnelles.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        logger.info(f"Analyse de contagion effectuée avec un seuil au {seuil_percentile}ème percentile")
        
        return proba_conditionnelles
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de contagion: {e}")
        return None

def analyser_similarite(df: pd.DataFrame, methode: str = 'euclidean', linkage_method: str = 'ward') -> Optional[Dict[str, Any]]:
    """
    Analyse la similarité entre les marchés.
    
    Args:
        df: DataFrame contenant les volatilités des pays
        methode: Méthode de calcul de la distance
        linkage_method: Méthode de liaison pour le clustering hiérarchique
        
    Returns:
        Dictionnaire contenant les résultats de l'analyse de similarité
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df is None or df.empty:
            logger.warning("Le DataFrame est vide ou None.")
            return None
        
        # Vérifier qu'il y a au moins deux pays
        if len(df.columns) < 2:
            logger.warning("Il faut au moins deux pays pour analyser la similarité.")
            return None
        
        # Calculer la matrice de distance
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        distance_matrix = pdist(df.T.values, metric=methode)
        distance_matrix_square = squareform(distance_matrix)
        
        # Convertir en DataFrame pour une meilleure lisibilité
        distance_df = pd.DataFrame(distance_matrix_square, index=df.columns, columns=df.columns)
        
        # Calculer le clustering hiérarchique
        Z = linkage(distance_matrix, method=linkage_method)
        
        # Créer un dictionnaire pour stocker les résultats
        resultats = {
            'distance_matrix': distance_df,
            'linkage': Z,
            'labels': df.columns
        }
        
        logger.info(f"Analyse de similarité effectuée avec la méthode de distance '{methode}' et la méthode de liaison '{linkage_method}'")
        
        return resultats
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de similarité: {e}")
        return None

def visualiser_correlation_dynamique(correlations: Dict[str, pd.Series], output_dir: str) -> None:
    """
    Visualise la corrélation dynamique entre les volatilités des différents pays.
    
    Args:
        correlations: Dictionnaire contenant les séries de corrélation dynamique
        output_dir: Répertoire de sortie pour les graphiques
    """
    try:
        # Vérifier que le dictionnaire n'est pas vide
        if not correlations:
            logger.warning("Le dictionnaire de corrélations est vide.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Créer un DataFrame pour stocker toutes les corrélations
        df_corr = pd.DataFrame()
        
        for paire, corr in correlations.items():
            df_corr[paire] = corr
        
        # Tracer le graphique
        plt.figure(figsize=(12, 8))
        
        for paire in df_corr.columns:
            plt.plot(df_corr.index, df_corr[paire], label=paire, linewidth=1.5)
        
        plt.title("Corrélation Dynamique entre les Volatilités des Pays")
        plt.xlabel('Date')
        plt.ylabel('Corrélation')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_file = os.path.join(output_dir, "correlation_dynamique.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Créer également une version interactive avec Plotly
        fig = go.Figure()
        
        for paire in df_corr.columns:
            fig.add_trace(go.Scatter(
                x=df_corr.index,
                y=df_corr[paire],
                mode='lines',
                name=paire
            ))
        
        fig.update_layout(
            title="Corrélation Dynamique entre les Volatilités des Pays",
            xaxis_title='Date',
            yaxis_title='Corrélation',
            template=VISUALISATION['interactivite']['template'],
            height=600,
            width=1000
        )
        
        # Sauvegarder la version interactive
        output_file_interactive = os.path.join(output_dir, "correlation_dynamique.html")
        fig.write_html(output_file_interactive)
        
        # Vérifier et convertir les données en valeurs numériques avant de tracer la heatmap
        corr_mean = df_corr.mean()
        pays = sorted(set([p.split('-')[0] for p in df_corr.columns] + [p.split('-')[1] for p in df_corr.columns]))
        corr_matrix = pd.DataFrame(index=pays, columns=pays)

        for paire, value in corr_mean.items():
            pays1, pays2 = paire.split('-')
            corr_matrix.loc[pays1, pays2] = value

        # Remplir la matrice symétrique et convertir les valeurs en numériques
        for i in corr_matrix.index:
            for j in corr_matrix.columns:
                if pd.isna(corr_matrix.loc[i, j]) and not pd.isna(corr_matrix.loc[j, i]):
                    corr_matrix.loc[i, j] = corr_matrix.loc[j, i]

        # Remplacer les NaN par 0 et s'assurer que toutes les valeurs sont numériques
        corr_matrix = corr_matrix.fillna(0).apply(pd.to_numeric, errors='coerce')
        
        # Tracer la heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                   fmt='.2f', square=True, linewidths=0.5, cbar_kws={'label': 'Corrélation moyenne'})
        plt.title("Corrélation Moyenne entre les Volatilités des Pays")
        plt.tight_layout()
        
        # Sauvegarder la heatmap
        output_file_heatmap = os.path.join(output_dir, "correlation_moyenne.png")
        plt.savefig(output_file_heatmap, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Créer également une version interactive avec Plotly
        fig_heatmap = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            text_auto='.2f'
        )
        
        fig_heatmap.update_layout(
            title="Corrélation Moyenne entre les Volatilités des Pays",
            xaxis_title="Pays",
            yaxis_title="Pays",
            template=VISUALISATION['interactivite']['template'],
            width=800,
            height=800
        )
        
        # Sauvegarder la version interactive
        output_file_heatmap_interactive = os.path.join(output_dir, "correlation_moyenne.html")
        fig_heatmap.write_html(output_file_heatmap_interactive)
        
        logger.info(f"Visualisation de la corrélation dynamique sauvegardée dans {output_file} et {output_file_interactive}")
        logger.info(f"Heatmap de la corrélation moyenne sauvegardée dans {output_file_heatmap} et {output_file_heatmap_interactive}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation de la corrélation dynamique: {e}")

def visualiser_contagion(proba_conditionnelles: pd.DataFrame, output_dir: str) -> None:
    """
    Visualise les probabilités conditionnelles de contagion.
    
    Args:
        proba_conditionnelles: DataFrame contenant les probabilités conditionnelles
        output_dir: Répertoire de sortie pour les graphiques
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if proba_conditionnelles is None or proba_conditionnelles.empty:
            logger.warning("Le DataFrame des probabilités conditionnelles est vide ou None.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Tracer la heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(proba_conditionnelles, annot=True, cmap='YlOrRd', vmin=0, vmax=1,
                   fmt='.2f', square=True, linewidths=0.5, cbar_kws={'label': 'Probabilité conditionnelle'})
        plt.title("Probabilités de Contagion entre les Pays")
        plt.xlabel("Pays affecté")
        plt.ylabel("Pays source")
        plt.tight_layout()
        
        # Sauvegarder la heatmap
        output_file = os.path.join(output_dir, "contagion.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Créer également une version interactive avec Plotly
        fig = px.imshow(
            proba_conditionnelles,
            x=proba_conditionnelles.columns,
            y=proba_conditionnelles.index,
            color_continuous_scale='YlOrRd',
            zmin=0,
            zmax=1,
            text_auto='.2f'
        )
        
        fig.update_layout(
            title="Probabilités de Contagion entre les Pays",
            xaxis_title="Pays affecté",
            yaxis_title="Pays source",
            template=VISUALISATION['interactivite']['template'],
            width=800,
            height=800
        )
        
        # Sauvegarder la version interactive
        output_file_interactive = os.path.join(output_dir, "contagion.html")
        fig.write_html(output_file_interactive)
        
        logger.info(f"Visualisation des probabilités de contagion sauvegardée dans {output_file} et {output_file_interactive}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation des probabilités de contagion: {e}")

def visualiser_similarite(resultats_similarite: Dict[str, Any], output_dir: str) -> None:
    """
    Visualise les résultats de l'analyse de similarité.
    
    Args:
        resultats_similarite: Dictionnaire contenant les résultats de l'analyse de similarité
        output_dir: Répertoire de sortie pour les graphiques
    """
    try:
        # Vérifier que le dictionnaire n'est pas vide
        if not resultats_similarite:
            logger.warning("Le dictionnaire des résultats de similarité est vide.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraire les données
        distance_df = resultats_similarite['distance_matrix']
        Z = resultats_similarite['linkage']
        labels = resultats_similarite['labels']
        
        # Tracer la heatmap de distance
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_df, annot=True, cmap='viridis', fmt='.2f', square=True,
                   linewidths=0.5, cbar_kws={'label': 'Distance'})
        plt.title("Matrice de Distance entre les Pays")
        plt.tight_layout()
        
        # Sauvegarder la heatmap
        output_file_heatmap = os.path.join(output_dir, "distance_matrix.png")
        plt.savefig(output_file_heatmap, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Tracer le dendrogramme
        plt.figure(figsize=(12, 8))
        dendrogram(Z, labels=labels, leaf_rotation=90)
        plt.title("Dendrogramme des Pays")
        plt.xlabel("Pays")
        plt.ylabel("Distance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Sauvegarder le dendrogramme
        output_file_dendro = os.path.join(output_dir, "dendrogramme.png")
        plt.savefig(output_file_dendro, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Créer également une version interactive de la heatmap avec Plotly
        fig_heatmap = px.imshow(
            distance_df,
            x=distance_df.columns,
            y=distance_df.index,
            color_continuous_scale='viridis',
            text_auto='.2f'
        )
        
        fig_heatmap.update_layout(
            title="Matrice de Distance entre les Pays",
            xaxis_title="Pays",
            yaxis_title="Pays",
            template=VISUALISATION['interactivite']['template'],
            width=800,
            height=800
        )
        
        # Sauvegarder la version interactive
        output_file_heatmap_interactive = os.path.join(output_dir, "distance_matrix.html")
        fig_heatmap.write_html(output_file_heatmap_interactive)
        
        logger.info(f"Heatmap de distance sauvegardée dans {output_file_heatmap} et {output_file_heatmap_interactive}")
        logger.info(f"Dendrogramme sauvegardé dans {output_file_dendro}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation des résultats de similarité: {e}")

def main():
    """
    Fonction principale.
    """
    try:
        # Charger les données de volatilité pour tous les pays
        volatilites = charger_donnees_volatilite_pays()
        
        if volatilites is None:
            logger.error("Impossible de charger les données de volatilité. Arrêt du script.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = CHEMINS['analyse_comparative']
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder les données de volatilité
        output_file_volatilites = os.path.join(output_dir, "volatilites_pays.csv")
        volatilites.to_csv(output_file_volatilites)
        
        logger.info(f"Données de volatilité sauvegardées dans {output_file_volatilites}")
        
        # Générer des statistiques descriptives
        stats = generer_statistiques_descriptives(volatilites)
        
        if not stats.empty:
            # Sauvegarder les statistiques descriptives
            output_file_stats = os.path.join(output_dir, "statistiques_descriptives.csv")
            stats.to_csv(output_file_stats)
            
            logger.info(f"Statistiques descriptives sauvegardées dans {output_file_stats}")
        
        # Analyser la corrélation dynamique
        fenetre = 60  # 60 jours (environ 3 mois)
        correlations = calculer_correlation_dynamique(volatilites, fenetre)
        
        if correlations:
            visualiser_correlation_dynamique(correlations, output_dir)
        
        # Analyser la contagion
        seuil_percentile = ANALYSE_COMPARATIVE['contagion']['seuil_percentile']
        proba_conditionnelles = analyser_contagion(volatilites, seuil_percentile)
        
        if proba_conditionnelles is not None:
            visualiser_contagion(proba_conditionnelles, output_dir)
            
            # Sauvegarder les probabilités conditionnelles
            output_file_proba = os.path.join(output_dir, "probabilites_contagion.csv")
            proba_conditionnelles.to_csv(output_file_proba)
            
            logger.info(f"Probabilités de contagion sauvegardées dans {output_file_proba}")
        
        # Analyser la similarité
        methode = ANALYSE_COMPARATIVE['similarite']['methode']
        linkage_method = ANALYSE_COMPARATIVE['similarite']['linkage']
        resultats_similarite = analyser_similarite(volatilites, methode, linkage_method)
        
        if resultats_similarite:
            visualiser_similarite(resultats_similarite, output_dir)
            
            # Sauvegarder la matrice de distance
            output_file_distance = os.path.join(output_dir, "matrice_distance.csv")
            resultats_similarite['distance_matrix'].to_csv(output_file_distance)
            
            logger.info(f"Matrice de distance sauvegardée dans {output_file_distance}")
        
        logger.info("Analyse comparative terminée avec succès")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")

if __name__ == "__main__":
    main()
