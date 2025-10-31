"""
Script d'analyse de l'impact des variables macroéconomiques sur la volatilité.
Version améliorée avec intégration des modules utils et config.
Version corrigée pour résoudre les problèmes de conversion de caractères spéciaux.
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
    CHEMINS, PAYS, ANALYSE_IMPACT, STATIONNARITE, 
    VALEURS_ABERRANTES, VISUALISATION
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CHEMINS.get('logs', ''), 'impact_macro.log'))
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def charger_donnees_volatilite(pays: str) -> Optional[pd.DataFrame]:
    """
    Charge les données de volatilité pour un pays.
    
    Args:
        pays: Clé du pays dans le dictionnaire PAYS
        
    Returns:
        DataFrame contenant les données de volatilité ou None en cas d'erreur
    """
    try:
        pays_dir = PAYS[pays]['dossier']
        file_path = os.path.join(CHEMINS['volatilite'], pays_dir, "resultats_volatilite.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Le fichier {file_path} n'existe pas.")
            return None
        
        # Charger les données
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Définir la colonne date comme index
        df = df.set_index('date')
        df = df.sort_index()
        
        logger.info(f"Données de volatilité chargées pour {PAYS[pays]['nom']}: {df.shape[0]} lignes x {df.shape[1]} colonnes")
        logger.info(f"Période: {df.index.min().strftime('%Y-%m-%d')} à {df.index.max().strftime('%Y-%m-%d')}")
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données de volatilité pour {PAYS[pays]['nom']}: {e}")
        return None

def charger_donnees_macro(pays: str) -> Optional[pd.DataFrame]:
    """
    Charge les données macroéconomiques pour un pays.
    
    Args:
        pays: Clé du pays dans le dictionnaire PAYS
        
    Returns:
        DataFrame contenant les données macroéconomiques ou None en cas d'erreur
    """
    try:
        pays_dir = PAYS[pays]['dossier']
        file_path = os.path.join(CHEMINS['data_harmonisee'], pays_dir, "donnees_fusionnees_final.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Le fichier {file_path} n'existe pas.")
            return None
        
        # Charger les données
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Définir la colonne date comme index
        df = df.set_index('date')
        df = df.sort_index()
        
        # Identifier les colonnes macroéconomiques
        macro_cols = ['inflation', 'taux_directeur', 'gdp', 'pib', 'close_usd', 'close_euro']
        macro_cols = [col for col in macro_cols if col in df.columns]
        
        if not macro_cols:
            logger.warning(f"Aucune colonne macroéconomique trouvée pour {PAYS[pays]['nom']}.")
            return None
        
        # Sélectionner uniquement les colonnes macroéconomiques
        df = df[macro_cols]
        
        # Standardiser les noms de colonnes (pib -> gdp)
        if 'pib' in df.columns and 'gdp' not in df.columns:
            df = df.rename(columns={'pib': 'gdp'})
        
        # Nettoyer les caractères non numériques et convertir en numérique
        for col in df.columns:
            if df[col].dtype == 'object':
                # Remplacer les espaces insécables et autres caractères problématiques
                df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Données macroéconomiques chargées pour {PAYS[pays]['nom']}: {df.shape[0]} lignes x {df.shape[1]} colonnes")
        logger.info(f"Colonnes disponibles: {', '.join(df.columns)}")
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données macroéconomiques pour {PAYS[pays]['nom']}: {e}")
        return None

def fusionner_donnees(df_volatilite: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les données de volatilité et macroéconomiques.
    
    Args:
        df_volatilite: DataFrame contenant les données de volatilité
        df_macro: DataFrame contenant les données macroéconomiques
        
    Returns:
        DataFrame fusionné
    """
    try:
        # Vérifier que les DataFrames ne sont pas vides
        if df_volatilite is None or df_volatilite.empty:
            logger.warning("Le DataFrame de volatilité est vide ou None.")
            return df_macro
        
        if df_macro is None or df_macro.empty:
            logger.warning("Le DataFrame macroéconomique est vide ou None.")
            return df_volatilite
        
        # Sélectionner les colonnes de volatilité
        volatility_cols = [col for col in df_volatilite.columns if col.startswith('volatilite_')]
        
        if not volatility_cols:
            logger.warning("Aucune colonne de volatilité trouvée.")
            return df_macro
        
        # Fusionner les DataFrames
        df = pd.merge(
            df_volatilite[volatility_cols],
            df_macro,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Vérifier que le DataFrame fusionné n'est pas vide
        if df.empty:
            logger.warning("Le DataFrame fusionné est vide. Vérifier les périodes des données.")
            return pd.DataFrame()
        
        # S'assurer que toutes les colonnes sont numériques
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Supprimer les lignes avec des valeurs manquantes
        df = df.dropna()
        
        logger.info(f"Données fusionnées: {df.shape[0]} lignes x {df.shape[1]} colonnes")
        logger.info(f"Période: {df.index.min().strftime('%Y-%m-%d')} à {df.index.max().strftime('%Y-%m-%d')}")
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors de la fusion des données: {e}")
        return pd.DataFrame()

def analyser_correlation(df: pd.DataFrame, pays: str) -> None:
    """
    Analyse la corrélation entre volatilité et variables macroéconomiques.
    
    Args:
        df: DataFrame contenant les données fusionnées
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df.empty:
            logger.warning("Le DataFrame est vide. Impossible d'analyser la corrélation.")
            return
        
        # Sélectionner les colonnes de volatilité
        volatility_cols = [col for col in df.columns if col.startswith('volatilite_')]
        
        if not volatility_cols:
            logger.warning("Aucune colonne de volatilité trouvée.")
            return
        
        # Sélectionner les colonnes macroéconomiques
        macro_cols = [col for col in df.columns if not col.startswith('volatilite_')]
        
        if not macro_cols:
            logger.warning("Aucune colonne macroéconomique trouvée.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        pays_dir = PAYS[pays]['dossier']
        output_dir = os.path.join(CHEMINS['analyse_impact'], pays_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Pour chaque méthode de corrélation
        for methode in ANALYSE_IMPACT['correlation']['methodes']:
            # Calculer la matrice de corrélation
            corr = df.corr(method=methode.lower())
            
            # Sélectionner uniquement les corrélations entre volatilité et variables macro
            corr_vol_macro = corr.loc[volatility_cols, macro_cols]
            
            # Tracer la matrice de corrélation
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_vol_macro, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                       fmt='.2f', linewidths=0.5)
            plt.title(f"Corrélation {methode.capitalize()} entre Volatilité et Variables Macroéconomiques - {PAYS[pays]['nom']}")
            plt.tight_layout()
            
            # Sauvegarder le graphique
            output_file = os.path.join(output_dir, f"correlation_{methode.lower()}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Créer également une version interactive avec Plotly
            fig = px.imshow(
                corr_vol_macro,
                x=corr_vol_macro.columns,
                y=corr_vol_macro.index,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                text_auto='.2f'
            )
            
            fig.update_layout(
                title=f"Corrélation {methode.capitalize()} entre Volatilité et Variables Macroéconomiques - {PAYS[pays]['nom']}",
                xaxis_title="Variables Macroéconomiques",
                yaxis_title="Volatilité",
                template=VISUALISATION['interactivite']['template'],
                width=900,
                height=600
            )
            
            # Sauvegarder la version interactive
            output_file_interactive = os.path.join(output_dir, f"correlation_{methode.lower()}.html")
            fig.write_html(output_file_interactive)
            
            logger.info(f"Matrice de corrélation {methode} sauvegardée dans {output_file} et {output_file_interactive}")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de corrélation: {e}")

def tester_causalite_granger(df: pd.DataFrame, pays: str) -> None:
    """
    Teste la causalité de Granger entre volatilité et variables macroéconomiques.
    
    Args:
        df: DataFrame contenant les données fusionnées
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df.empty:
            logger.warning("Le DataFrame est vide. Impossible de tester la causalité de Granger.")
            return
        
        # Sélectionner les colonnes de volatilité
        volatility_cols = [col for col in df.columns if col.startswith('volatilite_')]
        
        if not volatility_cols:
            logger.warning("Aucune colonne de volatilité trouvée.")
            return
        
        # Sélectionner les colonnes macroéconomiques
        macro_cols = [col for col in df.columns if not col.startswith('volatilite_')]
        
        if not macro_cols:
            logger.warning("Aucune colonne macroéconomique trouvée.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        pays_dir = PAYS[pays]['dossier']
        output_dir = os.path.join(CHEMINS['analyse_impact'], pays_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Récupérer les paramètres
        max_lag = ANALYSE_IMPACT['causalite_granger']['max_lag']
        seuil_significativite = ANALYSE_IMPACT['causalite_granger']['seuil_significativite']
        
        # Créer un DataFrame pour stocker les résultats
        resultats = pd.DataFrame(
            index=pd.MultiIndex.from_product([volatility_cols, macro_cols], names=['volatilite', 'variable_macro']),
            columns=[f'lag_{i}' for i in range(1, max_lag + 1)] + ['min_pvalue', 'lag_optimal', 'causalite']
        )
        
        # Pour chaque paire (volatilité, variable macro)
        for vol_col in volatility_cols:
            for macro_col in macro_cols:
                # Sélectionner les données
                data = df[[vol_col, macro_col]].dropna()
                
                if len(data) < max_lag + 10:  # Pas assez de données
                    logger.warning(f"Pas assez de données pour tester la causalité entre {vol_col} et {macro_col}.")
                    continue
                
                # Tester la stationnarité
                est_stationnaire_vol, _, _ = tester_stationnarite(data[vol_col], vol_col)
                est_stationnaire_macro, _, _ = tester_stationnarite(data[macro_col], macro_col)
                
                # Différencier si nécessaire
                if not est_stationnaire_vol:
                    try:
                        data[vol_col] = differencier_serie(data[vol_col])
                    except Exception as e:
                        logger.error(f"Erreur lors de la différenciation de la série {vol_col}: {e}")
                        continue
                
                if not est_stationnaire_macro:
                    try:
                        data[macro_col] = differencier_serie(data[macro_col])
                    except Exception as e:
                        logger.error(f"Erreur lors de la différenciation de la série {macro_col}: {e}")
                        continue
                
                # Supprimer les valeurs NaN après différenciation
                data = data.dropna()
                
                if len(data) < max_lag + 10:  # Pas assez de données après différenciation
                    logger.warning(f"Pas assez de données après différenciation pour tester la causalité entre {vol_col} et {macro_col}.")
                    continue
                
                # Tester la causalité de Granger dans les deux sens
                # 1. Variable macro -> Volatilité
                try:
                    gc_result_macro_to_vol = grangercausalitytests(data[[vol_col, macro_col]], maxlag=max_lag)
                    
                    # Extraire les p-values
                    p_values_macro_to_vol = [round(gc_result_macro_to_vol[i+1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]
                    
                    # Stocker les résultats
                    for i, p_value in enumerate(p_values_macro_to_vol):
                        resultats.loc[(vol_col, macro_col), f'lag_{i+1}'] = p_value
                    
                    # Trouver la p-value minimale et le lag correspondant
                    min_pvalue = min(p_values_macro_to_vol)
                    lag_optimal = p_values_macro_to_vol.index(min_pvalue) + 1
                    
                    resultats.loc[(vol_col, macro_col), 'min_pvalue'] = min_pvalue
                    resultats.loc[(vol_col, macro_col), 'lag_optimal'] = lag_optimal
                    resultats.loc[(vol_col, macro_col), 'causalite'] = min_pvalue < seuil_significativite
                    
                except Exception as e:
                    logger.warning(f"Erreur lors du test de causalité de Granger entre {vol_col} et {macro_col}: {e}")
        
        # Sauvegarder les résultats
        output_file = os.path.join(output_dir, "causalite_granger.csv")
        resultats.to_csv(output_file)
        
        # Créer un résumé des résultats significatifs
        resultats_significatifs = resultats[resultats['causalite'] == True]
        
        if not resultats_significatifs.empty:
            # Sauvegarder les résultats significatifs
            output_file_significatifs = os.path.join(output_dir, "causalite_granger_significatifs.csv")
            resultats_significatifs.to_csv(output_file_significatifs)
            
            # Créer une visualisation des résultats significatifs
            plt.figure(figsize=(12, 8))
            
            # Préparer les données pour la visualisation
            viz_data = resultats_significatifs.reset_index()
            viz_data['pair'] = viz_data['variable_macro'] + ' -> ' + viz_data['volatilite']
            
            # Tracer le graphique
            sns.barplot(x='min_pvalue', y='pair', data=viz_data)
            plt.axvline(x=seuil_significativite, color='red', linestyle='--', label=f'Seuil de significativité ({seuil_significativite})')
            plt.title(f"Relations de causalité significatives (Granger) - {PAYS[pays]['nom']}")
            plt.xlabel('p-value minimale')
            plt.ylabel('Relation de causalité')
            plt.legend()
            plt.tight_layout()
            
            # Sauvegarder le graphique
            output_file_viz = os.path.join(output_dir, "causalite_granger_viz.png")
            plt.savefig(output_file_viz, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Créer également une version interactive avec Plotly
            fig = px.bar(
                viz_data,
                x='min_pvalue',
                y='pair',
                color='lag_optimal',
                labels={'min_pvalue': 'p-value minimale', 'pair': 'Relation de causalité', 'lag_optimal': 'Lag optimal'},
                title=f"Relations de causalité significatives (Granger) - {PAYS[pays]['nom']}",
                orientation='h'
            )
            
            fig.add_vline(
                x=seuil_significativite,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Seuil de significativité ({seuil_significativite})"
            )
            
            fig.update_layout(
                template=VISUALISATION['interactivite']['template'],
                width=900,
                height=600
            )
            
            # Sauvegarder la version interactive
            output_file_viz_interactive = os.path.join(output_dir, "causalite_granger_viz.html")
            fig.write_html(output_file_viz_interactive)
            
            logger.info(f"Résultats du test de causalité de Granger sauvegardés dans {output_file}")
            logger.info(f"Résultats significatifs sauvegardés dans {output_file_significatifs}")
            logger.info(f"Visualisation sauvegardée dans {output_file_viz} et {output_file_viz_interactive}")
        else:
            logger.info(f"Aucune relation de causalité significative trouvée pour {PAYS[pays]['nom']}")
    
    except Exception as e:
        logger.error(f"Erreur lors du test de causalité de Granger: {e}")

def analyser_var(df: pd.DataFrame, pays: str) -> None:
    """
    Analyse les interactions avec un modèle VAR.
    
    Args:
        df: DataFrame contenant les données fusionnées
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df.empty:
            logger.warning("Le DataFrame est vide. Impossible d'analyser avec un modèle VAR.")
            return
        
        # Sélectionner les colonnes de volatilité
        volatility_cols = [col for col in df.columns if col.startswith('volatilite_')]
        
        if not volatility_cols:
            logger.warning("Aucune colonne de volatilité trouvée.")
            return
        
        # Sélectionner les colonnes macroéconomiques
        macro_cols = [col for col in df.columns if not col.startswith('volatilite_')]
        
        if not macro_cols:
            logger.warning("Aucune colonne macroéconomique trouvée.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        pays_dir = PAYS[pays]['dossier']
        output_dir = os.path.join(CHEMINS['analyse_impact'], pays_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Récupérer les paramètres
        max_lag = ANALYSE_IMPACT['var']['max_lag']
        selection_lag = ANALYSE_IMPACT['var']['selection_lag']
        
        # Pour chaque volatilité
        for vol_col in volatility_cols:
            # Sélectionner les variables pour le modèle VAR
            var_cols = [vol_col] + macro_cols
            
            # Sélectionner les données
            data = df[var_cols].dropna()
            
            if len(data) < max_lag + 10:  # Pas assez de données
                logger.warning(f"Pas assez de données pour ajuster un modèle VAR pour {vol_col}.")
                continue
            
            # Tester la stationnarité de chaque série
            for col in var_cols:
                try:
                    est_stationnaire, _, _ = tester_stationnarite(data[col], col)
                    
                    if not est_stationnaire:
                        data[col] = differencier_serie(data[col])
                except Exception as e:
                    logger.error(f"Erreur lors du test de stationnarité pour {col}: {e}")
                    # Remplacer par NaN en cas d'erreur
                    data[col] = np.nan
            
            # Supprimer les valeurs NaN après différenciation
            data = data.dropna()
            
            if len(data) < max_lag + 10:  # Pas assez de données après différenciation
                logger.warning(f"Pas assez de données après différenciation pour ajuster un modèle VAR pour {vol_col}.")
                continue
            
            # Vérifier que toutes les colonnes sont numériques
            for col in data.columns:
                if data[col].dtype == 'object':
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Erreur lors de la conversion en numérique pour {col}: {e}")
                        # Supprimer la colonne en cas d'erreur
                        data = data.drop(columns=[col])
            
            # Supprimer à nouveau les valeurs NaN après conversion
            data = data.dropna()
            
            if data.empty or len(data.columns) < 2:
                logger.warning(f"Pas assez de colonnes valides pour ajuster un modèle VAR pour {vol_col}.")
                continue
            
            # Ajuster le modèle VAR
            try:
                model = VAR(data)
                
                # Sélectionner le lag optimal
                if selection_lag == 'aic':
                    results = model.fit(maxlags=max_lag, ic='aic')
                elif selection_lag == 'bic':
                    results = model.fit(maxlags=max_lag, ic='bic')
                else:
                    results = model.fit(maxlags=max_lag)
                
                # Sauvegarder le résumé du modèle
                output_file_summary = os.path.join(output_dir, f"var_summary_{vol_col.split('_')[-1]}.txt")
                with open(output_file_summary, 'w') as f:
                    f.write(str(results.summary()))
                
                # Analyser les fonctions de réponse impulsionnelle
                irf = results.irf(10)  # 10 périodes
                
                # Tracer les fonctions de réponse impulsionnelle
                plt.figure(figsize=(15, 10))
                irf.plot(orth=False)
                plt.suptitle(f"Fonctions de Réponse Impulsionnelle - {vol_col} - {PAYS[pays]['nom']}")
                plt.tight_layout()
                
                # Sauvegarder le graphique
                output_file_irf = os.path.join(output_dir, f"var_irf_{vol_col.split('_')[-1]}.png")
                plt.savefig(output_file_irf, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Décomposition de la variance
                fevd = results.fevd(10)  # 10 périodes
                
                # Tracer la décomposition de la variance
                plt.figure(figsize=(15, 10))
                fevd.plot()
                plt.suptitle(f"Décomposition de la Variance - {vol_col} - {PAYS[pays]['nom']}")
                plt.tight_layout()
                
                # Sauvegarder le graphique
                output_file_fevd = os.path.join(output_dir, f"var_fevd_{vol_col.split('_')[-1]}.png")
                plt.savefig(output_file_fevd, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Modèle VAR ajusté pour {vol_col} avec {len(data)} observations et {results.k_ar} lags")
                logger.info(f"Résumé sauvegardé dans {output_file_summary}")
                logger.info(f"Fonctions de réponse impulsionnelle sauvegardées dans {output_file_irf}")
                logger.info(f"Décomposition de la variance sauvegardée dans {output_file_fevd}")
                
            except Exception as e:
                logger.warning(f"Erreur lors de l'ajustement du modèle VAR pour {vol_col}: {e}")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse VAR: {e}")

def tester_cointegration(df: pd.DataFrame, pays: str) -> None:
    """
    Teste la cointégration entre volatilité et variables macroéconomiques.
    
    Args:
        df: DataFrame contenant les données fusionnées
        pays: Clé du pays dans le dictionnaire PAYS
    """
    try:
        # Vérifier que le DataFrame n'est pas vide
        if df.empty:
            logger.warning("Le DataFrame est vide. Impossible de tester la cointégration.")
            return
        
        # Sélectionner les colonnes de volatilité
        volatility_cols = [col for col in df.columns if col.startswith('volatilite_')]
        
        if not volatility_cols:
            logger.warning("Aucune colonne de volatilité trouvée.")
            return
        
        # Sélectionner les colonnes macroéconomiques
        macro_cols = [col for col in df.columns if not col.startswith('volatilite_')]
        
        if not macro_cols:
            logger.warning("Aucune colonne macroéconomique trouvée.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        pays_dir = PAYS[pays]['dossier']
        output_dir = os.path.join(CHEMINS['analyse_impact'], pays_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Créer un DataFrame pour stocker les résultats
        resultats = pd.DataFrame(
            index=pd.MultiIndex.from_product([volatility_cols, macro_cols], names=['volatilite', 'variable_macro']),
            columns=['t_stat', 'p_value', 'cointegration']
        )
        
        # Pour chaque paire (volatilité, variable macro)
        for vol_col in volatility_cols:
            for macro_col in macro_cols:
                # Sélectionner les données
                data = df[[vol_col, macro_col]].dropna()
                
                if len(data) < 30:  # Pas assez de données
                    logger.warning(f"Pas assez de données pour tester la cointégration entre {vol_col} et {macro_col}.")
                    continue
                
                # S'assurer que les données sont numériques
                try:
                    data[vol_col] = pd.to_numeric(data[vol_col], errors='coerce')
                    data[macro_col] = pd.to_numeric(data[macro_col], errors='coerce')
                    data = data.dropna()
                    
                    if len(data) < 30:  # Vérifier à nouveau après conversion
                        logger.warning(f"Pas assez de données après conversion pour tester la cointégration entre {vol_col} et {macro_col}.")
                        continue
                    
                    # Tester la cointégration
                    result = coint(data[vol_col], data[macro_col])
                    
                    # Extraire les résultats
                    t_stat, p_value, _ = result
                    
                    # Stocker les résultats
                    resultats.loc[(vol_col, macro_col), 't_stat'] = t_stat
                    resultats.loc[(vol_col, macro_col), 'p_value'] = p_value
                    resultats.loc[(vol_col, macro_col), 'cointegration'] = p_value < 0.05
                    
                except Exception as e:
                    logger.warning(f"Erreur lors du test de cointégration entre {vol_col} et {macro_col}: {e}")
        
        # Sauvegarder les résultats
        output_file = os.path.join(output_dir, "cointegration.csv")
        resultats.to_csv(output_file)
        
        # Créer un résumé des résultats significatifs
        resultats_significatifs = resultats[resultats['cointegration'] == True]
        
        if not resultats_significatifs.empty:
            # Sauvegarder les résultats significatifs
            output_file_significatifs = os.path.join(output_dir, "cointegration_significatifs.csv")
            resultats_significatifs.to_csv(output_file_significatifs)
            
            # Créer une visualisation des résultats significatifs
            plt.figure(figsize=(12, 8))
            
            # Préparer les données pour la visualisation
            viz_data = resultats_significatifs.reset_index()
            viz_data['pair'] = viz_data['volatilite'] + ' - ' + viz_data['variable_macro']
            
            # Tracer le graphique
            sns.barplot(x='p_value', y='pair', data=viz_data)
            plt.axvline(x=0.05, color='red', linestyle='--', label='Seuil de significativité (0.05)')
            plt.title(f"Relations de cointégration significatives - {PAYS[pays]['nom']}")
            plt.xlabel('p-value')
            plt.ylabel('Paire de variables')
            plt.legend()
            plt.tight_layout()
            
            # Sauvegarder le graphique
            output_file_viz = os.path.join(output_dir, "cointegration_viz.png")
            plt.savefig(output_file_viz, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Créer également une version interactive avec Plotly
            fig = px.bar(
                viz_data,
                x='p_value',
                y='pair',
                labels={'p_value': 'p-value', 'pair': 'Paire de variables'},
                title=f"Relations de cointégration significatives - {PAYS[pays]['nom']}",
                orientation='h'
            )
            
            fig.add_vline(
                x=0.05,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil de significativité (0.05)"
            )
            
            fig.update_layout(
                template=VISUALISATION['interactivite']['template'],
                width=900,
                height=600
            )
            
            # Sauvegarder la version interactive
            output_file_viz_interactive = os.path.join(output_dir, "cointegration_viz.html")
            fig.write_html(output_file_viz_interactive)
            
            logger.info(f"Résultats du test de cointégration sauvegardés dans {output_file}")
            logger.info(f"Résultats significatifs sauvegardés dans {output_file_significatifs}")
            logger.info(f"Visualisation sauvegardée dans {output_file_viz} et {output_file_viz_interactive}")
        else:
            logger.info(f"Aucune relation de cointégration significative trouvée pour {PAYS[pays]['nom']}")
    
    except Exception as e:
        logger.error(f"Erreur lors du test de cointégration: {e}")

def main():
    """
    Fonction principale pour l'analyse de l'impact des variables macroéconomiques.
    """
    logger.info("Début de l'analyse de l'impact des variables macroéconomiques.")
    
    # Pour chaque pays
    for pays_key, pays_info in PAYS.items():
        logger.info(f"Traitement du pays: {pays_key}")
        
        # Charger les données de volatilité
        df_volatilite = charger_donnees_volatilite(pays_key)
        
        # Charger les données macroéconomiques
        df_macro = charger_donnees_macro(pays_key)
        
        # Fusionner les données
        df = fusionner_donnees(df_volatilite, df_macro)
        
        if df.empty:
            logger.warning(f"Aucune donnée fusionnée disponible pour {pays_info['nom']}. Passage au pays suivant.")
            continue
        
        # Analyser la corrélation
        analyser_correlation(df, pays_key)
        
        # Tester la causalité de Granger
        tester_causalite_granger(df, pays_key)
        
        # Analyser avec un modèle VAR
        analyser_var(df, pays_key)
        
        # Tester la cointégration
        tester_cointegration(df, pays_key)
        
        logger.info(f"Traitement terminé pour {pays_key}")
    
    logger.info("Traitement terminé pour tous les pays")

if __name__ == "__main__":
    main()
