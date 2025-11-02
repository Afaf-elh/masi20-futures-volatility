"""
Script de visualisation des erreurs de prédiction pour les différents pays.
Ce script génère des visualisations interactives pour analyser la distribution
des erreurs de prédiction des modèles de volatilité.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import logging
from typing import Optional, List, Dict, Any

# Importer les modules utilitaires et de configuration
from utils import charger_donnees, sauvegarder_donnees
from config import CHEMINS, PAYS, VISUALISATION

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CHEMINS.get('logs', ''), 'visualisation_erreurs.log'))
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def sauvegarder_figures(fig, output_dir: str, base_filename: str, pays: str, width: int = 1000, height: int = 600):
    """
    Sauvegarde les figures en format HTML et PNG de manière robuste.
    
    Args:
        fig: Figure Plotly à sauvegarder
        output_dir: Répertoire de sortie
        base_filename: Nom de base du fichier
        pays: Code du pays
        width: Largeur de l'image (défaut: 1000)
        height: Hauteur de l'image (défaut: 600)
    """
    try:
        # Sauvegarder en HTML (toujours possible)
        output_file_html = os.path.join(output_dir, f"{base_filename}_{pays}.html")
        fig.write_html(output_file_html)
        logger.info(f"Figure HTML sauvegardée: {output_file_html}")
        
        # 🔥 CORRECTION : Essayer PNG avec gestion d'erreur
        try:
            output_file_png = os.path.join(output_dir, f"{base_filename}_{pays}.png")
            fig.write_image(output_file_png, width=width, height=height, scale=2)
            logger.info(f"Figure PNG sauvegardée: {output_file_png}")
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder PNG pour {pays}: {e}")
            
    except Exception as e:
        logger.error(f"Erreur sauvegarde figures pour {pays}: {e}")
        
def charger_donnees_erreurs(pays: str) -> Optional[pd.DataFrame]:
    """
    Charge les données d'erreurs de prédiction pour un pays.
    
    Args:
        pays: Clé du pays dans le dictionnaire PAYS
        
    Returns:
        DataFrame contenant les erreurs de prédiction ou None en cas d'erreur
    """
    try:
        pays_dir = PAYS[pays]['dossier']
        file_path = os.path.join(CHEMINS['rapport_final'], 'resultats', f"erreurs_prediction_{pays}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Le fichier {file_path} n'existe pas.")
            return None
        
        # Charger les données
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Définir la colonne date comme index
        df = df.set_index('date')
        df = df.sort_index()
        
        # Vérifier la structure des colonnes et s'adapter
        colonnes_existantes = df.columns.tolist()
        logger.info(f"Colonnes disponibles pour {pays}: {colonnes_existantes}")
        
        # Mapping des colonnes selon ce qui existe
        mapping_colonnes = {}
        
        # Colonnes d'erreurs
        noms_modeles = ['random_forest', 'xgboost', 'lightgbm', 'neural_network', 'lstm', 'cnn_lstm', 'Ensemble']
        for nom in noms_modeles:
            if nom in colonnes_existantes:
                mapping_colonnes[nom] = f'erreur_{nom}'
            
        # Colonne de volatilité réelle (si elle existe)
        if 'volatilite_reelle' in colonnes_existantes:
            mapping_colonnes['volatilite_reelle'] = 'volatilite_reelle'
        elif 'realized_volatility' in colonnes_existantes:
            mapping_colonnes['realized_volatility'] = 'volatilite_reelle'
        elif 'true_volatility' in colonnes_existantes:
            mapping_colonnes['true_volatility'] = 'volatilite_reelle'
            
        # Colonnes de prédictions (si elles existent)
        for col in colonnes_existantes:
            if col.startswith('prediction_'):
                # Garder le même nom sans ajouter un deuxième préfixe
                mapping_colonnes[col] = col
        
        # Appliquer le renommage seulement pour les colonnes qui existent
        colonnes_a_renommer = {k: v for k, v in mapping_colonnes.items() if k in colonnes_existantes}
        df.rename(columns=colonnes_a_renommer, inplace=True)
        
        logger.info(f"Données d'erreurs chargées pour {PAYS[pays]['nom']}: {df.shape[0]} lignes x {df.shape[1]} colonnes")
        logger.info(f"Colonnes après renommage: {df.columns.tolist()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données d'erreurs pour {PAYS[pays]['nom']}: {e}")
        return None

def visualiser_distribution_erreurs(df: pd.DataFrame, pays: str) -> None:
    """
    Visualise la distribution des erreurs de prédiction.
    
    Args:
        df: DataFrame contenant les erreurs de prédiction
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df is None or df.empty:
            logger.warning(f"Le DataFrame des erreurs pour {pays} est vide ou None.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = os.path.join(CHEMINS['rapport_final'], 'visualisations')
        os.makedirs(output_dir, exist_ok=True)
        
        # Identifier les colonnes d'erreurs
        error_cols = [col for col in df.columns if col.startswith('erreur_')]
        
        if not error_cols:
            logger.warning(f"Aucune colonne d'erreur trouvée pour {pays}.")
            return
        
        # Créer un DataFrame long pour faciliter la visualisation
        df_long = pd.melt(df.reset_index(), 
                          id_vars=['date'], 
                          value_vars=error_cols,
                          var_name='modele', 
                          value_name='erreur')
        
        # Nettoyer les noms des modèles
        df_long['modele'] = df_long['modele'].str.replace('erreur_', '').str.replace('_', ' ').str.title()
        
        # Créer la figure avec Plotly
        fig = px.histogram(
            df_long, 
            x='erreur', 
            color='modele',
            marginal='box',
            barmode='overlay',
            opacity=0.7,
            nbins=50,
            title=f"Distribution des erreurs de prédiction - {PAYS[pays]['nom']}",
            labels={'erreur': 'Erreur de prédiction', 'modele': 'Modèle'}
        )
        
        fig.update_layout(
            template=VISUALISATION['interactivite']['template'],
            height=600,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Ajouter une ligne verticale à zéro
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
            annotation_text="Erreur nulle"
        )
        
        # Sauvegarder la figure
        sauvegarder_figures(fig, output_dir, "distribution_erreurs", pays)

    except Exception as e:
        logger.error(f"Erreur lors de la visualisation de la distribution des erreurs pour {PAYS[pays]['nom']}: {e}")

def visualiser_erreurs_temporelles(df: pd.DataFrame, pays: str) -> None:
    """
    Visualise l'évolution temporelle des erreurs de prédiction.
    
    Args:
        df: DataFrame contenant les erreurs de prédiction
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df is None or df.empty:
            logger.warning(f"Le DataFrame des erreurs pour {pays} est vide ou None.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = os.path.join(CHEMINS['rapport_final'], 'visualisations')
        os.makedirs(output_dir, exist_ok=True)
        
        # Identifier les colonnes d'erreurs
        error_cols = [col for col in df.columns if col.startswith('erreur_')]
        
        if not error_cols:
            logger.warning(f"Aucune colonne d'erreur trouvée pour {pays}.")
            return
        
        # Créer la figure avec Plotly
        fig = go.Figure()
        
        for col in error_cols:
            nom_modele = col.replace('erreur_', '').replace('_', ' ').title()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=nom_modele
            ))
        
        # Ajouter une ligne horizontale à zéro
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
            annotation_text="Erreur nulle"
        )
        
        # Ajouter des annotations pour les événements importants si disponibles
        if VISUALISATION['annotations'] and pays in VISUALISATION['evenements']:
            evenements = VISUALISATION['evenements'][pays]
            
            for date_str, description in evenements.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                
                # Vérifier si la date est dans l'index
                if date in df.index:
                    # Trouver la valeur moyenne des erreurs à cette date
                    valeur = df.loc[date, error_cols].mean()
                    
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
            title=f"Évolution temporelle des erreurs de prédiction - {PAYS[pays]['nom']}",
            xaxis_title='Date',
            yaxis_title='Erreur de prédiction',
            template=VISUALISATION['interactivite']['template'],
            height=600,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Sauvegarder la figure
        sauvegarder_figures(fig, output_dir, "erreurs_temporelles", pays)

    except Exception as e:
        logger.error(f"Erreur lors de la visualisation de l'évolution temporelle des erreurs pour {PAYS[pays]['nom']}: {e}")

def visualiser_comparaison_predictions(df: pd.DataFrame, pays: str) -> None:
    """
    Visualise la comparaison entre les prédictions et les valeurs réelles.
    
    Args:
        df: DataFrame contenant les erreurs de prédiction
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df is None or df.empty:
            logger.warning(f"Le DataFrame des erreurs pour {pays} est vide ou None.")
            return
        
        # Vérifier que les colonnes nécessaires existent
        if 'volatilite_reelle' not in df.columns:
            logger.warning(f"La colonne 'volatilite_reelle' n'existe pas pour {pays}.")
            return
        
        # Identifier les colonnes de prédiction
        pred_cols = [col for col in df.columns if col.startswith('prediction_')]
        
        if not pred_cols:
            logger.warning(f"Aucune colonne de prédiction trouvée pour {pays}.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = os.path.join(CHEMINS['rapport_final'], 'visualisations')
        os.makedirs(output_dir, exist_ok=True)
        
        # Créer la figure avec Plotly
        fig = go.Figure()
        
        # Ajouter la volatilité réelle
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['volatilite_reelle'],
            mode='lines',
            name='Volatilité Réelle',
            line=dict(color='black', width=2)
        ))
        
        # Ajouter les prédictions
        for col in pred_cols:
            nom_modele = col.replace('prediction_', '').replace('_', ' ').title()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=f'Prédiction {nom_modele}',
                line=dict(dash='dash')
            ))
        
        # Ajouter des annotations pour les événements importants si disponibles
        if VISUALISATION['annotations'] and pays in VISUALISATION['evenements']:
            evenements = VISUALISATION['evenements'][pays]
            
            for date_str, description in evenements.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                
                # Vérifier si la date est dans l'index
                if date in df.index:
                    # Trouver la valeur de volatilité réelle à cette date
                    valeur = df.loc[date, 'volatilite_reelle']
                    
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
            title=f"Comparaison des prédictions de volatilité - {PAYS[pays]['nom']}",
            xaxis_title='Date',
            yaxis_title='Volatilité',
            template=VISUALISATION['interactivite']['template'],
            height=600,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Sauvegarder la figure
        sauvegarder_figures(fig, output_dir, "comparaison_predictions", pays)

    
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation de la comparaison des prédictions pour {PAYS[pays]['nom']}: {e}")

def visualiser_metriques_performance(df: pd.DataFrame, pays: str) -> None:
    """
    Visualise les métriques de performance des modèles.
    
    Args:
        df: DataFrame contenant les erreurs de prédiction
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df is None or df.empty:
            logger.warning(f"Le DataFrame des erreurs pour {pays} est vide ou None.")
            return
        
        # Vérifier que les colonnes nécessaires existent
        if 'volatilite_reelle' not in df.columns:
            logger.warning(f"La colonne 'volatilite_reelle' n'existe pas pour {pays}.")
            return
        
        # Identifier les colonnes de prédiction
        pred_cols = [col for col in df.columns if col.startswith('prediction_')]
        
        if not pred_cols:
            logger.warning(f"Aucune colonne de prédiction trouvée pour {pays}.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = os.path.join(CHEMINS['rapport_final'], 'visualisations')
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculer les métriques de performance
        metriques = pd.DataFrame(index=[col.replace('prediction_', '').replace('_', ' ').title() for col in pred_cols],
                                columns=['MAE', 'MSE', 'RMSE', 'MAPE'])
        
        for i, col in enumerate(pred_cols):
            nom_modele = col.replace('prediction_', '').replace('_', ' ').title()
            
            # Calculer les erreurs
            erreurs = df['volatilite_reelle'] - df[col]
            erreurs_abs = np.abs(erreurs)
            erreurs_carres = erreurs ** 2
            erreurs_pct = erreurs_abs / df['volatilite_reelle'] * 100
            
            # Calculer les métriques
            mae = erreurs_abs.mean()
            mse = erreurs_carres.mean()
            rmse = np.sqrt(mse)
            mape = erreurs_pct.mean()
            
            # Stocker les métriques
            metriques.loc[nom_modele, 'MAE'] = mae
            metriques.loc[nom_modele, 'MSE'] = mse
            metriques.loc[nom_modele, 'RMSE'] = rmse
            metriques.loc[nom_modele, 'MAPE'] = mape
        
        # Sauvegarder les métriques
        output_file_csv = os.path.join(output_dir, f"metriques_performance_{pays}.csv")
        metriques.to_csv(output_file_csv)
        
        # Créer un graphique pour chaque métrique
        for metrique in ['MAE', 'RMSE', 'MAPE']:
            fig = px.bar(
                metriques.sort_values(metrique),
                y=metriques.index,
                x=metrique,
                orientation='h',
                title=f"{metrique} par modèle - {PAYS[pays]['nom']}",
                labels={metrique: metrique, 'index': 'Modèle'},
                color=metriques.index,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            fig.update_layout(
                template=VISUALISATION['interactivite']['template'],
                height=500,
                width=900,
                showlegend=False
            )
            
            # Sauvegarder la figure
            sauvegarder_figures(fig, output_dir, metrique.lower(), pays, 900, 500)


        # Créer un tableau récapitulatif
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Modèle'] + list(metriques.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[metriques.index] + [metriques[col].round(4) for col in metriques.columns],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(
            title=f"Métriques de performance par modèle - {PAYS[pays]['nom']}",
            height=400,
            width=900
        )
        
        # Sauvegarder la figure
        sauvegarder_figures(fig, output_dir, "tableau_metriques", pays, 900, 400)
        
        logger.info(f"Tableau des métriques pour {PAYS[pays]['nom']} sauvegardé dans {sauvegarder_figures}")
        logger.info(f"Métriques de performance sauvegardées dans {output_file_csv}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation des métriques de performance pour {PAYS[pays]['nom']}: {e}")

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
            
            # Charger les données d'erreurs
            df = charger_donnees_erreurs(pays)
            
            if df is None:
                logger.warning(f"Impossible de charger les données d'erreurs pour {pays}. Passage au pays suivant.")
                continue
            
            # Visualiser la distribution des erreurs
            visualiser_distribution_erreurs(df, pays)
            
            # Visualiser l'évolution temporelle des erreurs
            visualiser_erreurs_temporelles(df, pays)
            
            if 'volatilite_reelle' in df.columns:
                visualiser_comparaison_predictions(df, pays)
                visualiser_metriques_performance(df, pays)
            else:
                logger.warning(f"Visualisations avancées ignorées pour {pays} (volatilite_reelle manquante)")
        
        logger.info("Traitement terminé pour tous les pays")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")

if __name__ == "__main__":
    main()
