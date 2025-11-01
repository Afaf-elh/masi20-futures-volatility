"""
Module d'utilitaires pour le projet de prédiction de volatilité des futures sur MASI20.
Ce module fournit des fonctions utilitaires utilisées par les différents scripts du projet.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import re
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import PAYS, CHEMINS

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("volatilite_futures.log")]
)
logger = logging.getLogger(__name__)

# Ignorer les avertissements
warnings.filterwarnings('ignore')

def installer_dependances():
    """
    Installe les dépendances nécessaires pour le projet.
    """
    try:
        import subprocess
        # Liste des packages nécessaires
        packages = [
            "openpyxl", "pandas", "numpy", "matplotlib", "seaborn", 
            "statsmodels", "scikit-learn", "plotly", "dash", "arch"
        ]
        
        for package in packages:
            try:
                # Vérifier si le package est déjà installé
                __import__(package)
                logger.info(f"Le package {package} est déjà installé.")
            except ImportError:
                # Installer le package s'il n'est pas déjà installé
                logger.info(f"Installation du package {package}...")
                subprocess.check_call(["pip", "install", package])
                logger.info(f"Package {package} installé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'installation des dépendances: {e}")

def charger_donnees(chemin_fichier: str) -> Optional[pd.DataFrame]:
    """
    Charge les données à partir d'un fichier CSV ou Excel.
    
    Args:
        chemin_fichier: Chemin du fichier à charger
        
    Returns:
        DataFrame contenant les données ou None en cas d'erreur
    """
    try:
        # Vérifier que le fichier existe
        if not os.path.exists(chemin_fichier):
            logger.error(f"Le fichier {chemin_fichier} n'existe pas.")
            return None
        
        # Déterminer le type de fichier
        extension = os.path.splitext(chemin_fichier)[1].lower()
        
        if extension == '.csv':
            # Essayer différents encodages et séparateurs
            try:
                df = pd.read_csv(chemin_fichier, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(chemin_fichier, encoding='latin-1')
                except:
                    try:
                        df = pd.read_csv(chemin_fichier, encoding='utf-8', sep=';')
                    except:
                        df = pd.read_csv(chemin_fichier, encoding='latin-1', sep=';')
        
        elif extension in ['.xlsx', '.xls']:
            # Installer openpyxl si nécessaire
            try:
                import openpyxl
            except ImportError:
                logger.info("Installation du package openpyxl...")
                import subprocess
                subprocess.check_call(["pip", "install", "openpyxl"])
                logger.info("Package openpyxl installé avec succès.")
            
            # Charger le fichier Excel
            df = pd.read_excel(chemin_fichier)
        
        else:
            logger.error(f"Format de fichier non pris en charge: {extension}")
            return None
        
        # Vérifier que le DataFrame n'est pas vide
        if df.empty:
            logger.warning(f"Le fichier {chemin_fichier} est vide.")
            return None
        
        logger.info(f"Données chargées avec succès depuis {chemin_fichier}: {df.shape[0]} lignes x {df.shape[1]} colonnes")
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier {chemin_fichier}: {e}")
        return None

def sauvegarder_donnees(df: pd.DataFrame, chemin_fichier: str) -> bool:
    """
    Sauvegarde les données dans un fichier CSV.
    
    Args:
        df: DataFrame à sauvegarder
        chemin_fichier: Chemin du fichier de sortie
        
    Returns:
        True si la sauvegarde a réussi, False sinon
    """
    try:
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)
        
        # Sauvegarder les données
        df.to_csv(chemin_fichier, index=False)
        
        logger.info(f"Données sauvegardées avec succès dans {chemin_fichier}")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données: {e}")
        return False

def standardiser_format_date(df: pd.DataFrame, colonne_date: str) -> pd.DataFrame:
    """
    Standardise le format de date dans un DataFrame vers le format AAAA-MM-JJ.
    
    Args:
        df: DataFrame à traiter
        colonne_date: Nom de la colonne contenant les dates
    
    Returns:
        DataFrame avec la colonne de date standardisée et utilisée comme index
    """
    try:
        df_copy = df.copy()
        
        # Vérifier si la colonne existe
        if colonne_date not in df_copy.columns:
            logger.warning(f"La colonne '{colonne_date}' n'existe pas dans le DataFrame.")
            # Essayer de trouver une colonne de date
            date_cols = [col for col in df_copy.columns if 'date' in col.lower() or 'jour' in col.lower() or 'time' in col.lower()]
            if date_cols:
                colonne_date = date_cols[0]
                logger.info(f"Utilisation de la colonne '{colonne_date}' comme colonne de date.")
            else:
                logger.error("Aucune colonne de date trouvée.")
                return df
        
        # Si la colonne est de type object, retirer d'éventuels guillemets et espaces
        if df_copy[colonne_date].dtype == object:
            # Supprimer les guillemets et les espaces
            df_copy[colonne_date] = df_copy[colonne_date].astype(str).str.replace('"', '', regex=False).str.strip()
        
        # Convertir la colonne en datetime
        df_copy[colonne_date] = pd.to_datetime(df_copy[colonne_date], errors='coerce')
        
        # Supprimer les lignes avec des dates NaT
        df_copy = df_copy.dropna(subset=[colonne_date])
        
        if len(df_copy) == 0:
            logger.warning(f"Attention: Aucune date valide trouvée dans la colonne '{colonne_date}'")
            return df
        
        # Définir la colonne de date comme index
        df_copy = df_copy.set_index(colonne_date)
        
        # Trier l'index par ordre croissant
        df_copy = df_copy.sort_index()
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors de la standardisation du format de date: {e}")
        return df

def convertir_frequence_journaliere(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit un DataFrame à une fréquence journalière.
    
    Args:
        df: DataFrame à convertir
        
    Returns:
        DataFrame converti à la fréquence journalière
    """
    try:
        # Créer une copie pour éviter de modifier l'original
        df_copy = df.copy()
        
        # Convertir les colonnes numériques en float
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                try:
                    # Essayer de convertir en float (retirer les séparateurs de milliers)
                    df_copy[col] = pd.to_numeric(df_copy[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
                except Exception:
                    # Si la conversion échoue, garder la colonne telle quelle
                    continue
        
        # Vérifier s'il y a au moins une colonne numérique
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logger.warning("Aucune colonne numérique trouvée pour l'interpolation")
            return df_copy
        
        # Rééchantillonner à la fréquence journalière
        df_daily = df_copy.resample('D').interpolate(method='linear')
        
        # Remplir les valeurs manquantes restantes
        df_daily = df_daily.fillna(method='ffill').fillna(method='bfill')
        
        return df_daily
    
    except Exception as e:
        logger.error(f"Erreur lors de la conversion à la fréquence journalière: {e}")
        return df

def calculer_rendements(df: pd.DataFrame, colonne: str = 'close', methode: str = 'log') -> pd.DataFrame:
    """
    Calcule les rendements journaliers à partir d'une série de prix.
    
    Args:
        df: DataFrame contenant les prix
        colonne: Nom de la colonne contenant les prix
        methode: Méthode de calcul des rendements ('log' ou 'simple')
        
    Returns:
        DataFrame avec une colonne supplémentaire pour les rendements
    """
    try:
        # Vérifier que la colonne existe
        if colonne not in df.columns:
            logger.warning(f"La colonne {colonne} n'existe pas dans le DataFrame.")
            return df
        
        # Calculer les rendements
        df_copy = df.copy()
        
        if methode == 'log':
            # Rendements logarithmiques
            df_copy['rendement'] = np.log(df_copy[colonne] / df_copy[colonne].shift(1)) * 100
        else:
            # Rendements simples
            df_copy['rendement'] = df_copy[colonne].pct_change() * 100
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des rendements: {e}")
        return df

def calculer_volatilite_historique(df: pd.DataFrame, colonne: str = 'rendement', fenetre: int = 30, annualisation: bool = True) -> pd.DataFrame:
    """
    Calcule la volatilité historique à partir d'une série de rendements.
    
    Args:
        df: DataFrame contenant les rendements
        colonne: Nom de la colonne contenant les rendements
        fenetre: Taille de la fenêtre mobile pour le calcul de la volatilité
        annualisation: Si True, annualise la volatilité
        
    Returns:
        DataFrame avec une colonne supplémentaire pour la volatilité
    """
    try:
        # Vérifier que la colonne existe
        if colonne not in df.columns:
            logger.warning(f"La colonne {colonne} n'existe pas dans le DataFrame.")
            return df
        
        # Calculer la volatilité
        df_copy = df.copy()
        df_copy['volatilite_historique'] = df_copy[colonne].rolling(window=fenetre).std()
        
        # Annualiser la volatilité si demandé
        if annualisation:
            df_copy['volatilite_historique'] = df_copy['volatilite_historique'] * np.sqrt(252)
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la volatilité: {e}")
        return df

def detecter_valeurs_aberrantes(df: pd.DataFrame, colonne: str, methode: str = 'zscore', seuil: float = 3.0, traitement: str = 'marquer') -> pd.DataFrame:
    """
    Détecte les valeurs aberrantes dans une série.
    
    Args:
        df: DataFrame contenant les données
        colonne: Nom de la colonne à analyser
        methode: Méthode de détection ('zscore' ou 'iqr')
        seuil: Seuil pour la détection
        traitement: Méthode de traitement ('marquer', 'remplacer', 'supprimer')
        
    Returns:
        DataFrame avec une colonne supplémentaire indiquant les valeurs aberrantes
    """
    try:
        # Vérifier que la colonne existe
        if colonne not in df.columns:
            logger.warning(f"La colonne {colonne} n'existe pas dans le DataFrame.")
            return df
        
        # Créer une copie du DataFrame
        df_copy = df.copy()
        
        # Détecter les valeurs aberrantes selon la méthode choisie
        if methode == 'zscore':
            # Méthode du Z-score
            z_scores = (df_copy[colonne] - df_copy[colonne].mean()) / df_copy[colonne].std()
            df_copy['aberrante'] = abs(z_scores) > seuil
        
        elif methode == 'iqr':
            # Méthode de l'écart interquartile
            Q1 = df_copy[colonne].quantile(0.25)
            Q3 = df_copy[colonne].quantile(0.75)
            IQR = Q3 - Q1
            df_copy['aberrante'] = (df_copy[colonne] < (Q1 - seuil * IQR)) | (df_copy[colonne] > (Q3 + seuil * IQR))
        
        else:
            logger.warning(f"Méthode {methode} non reconnue. Utilisation du Z-score.")
            z_scores = (df_copy[colonne] - df_copy[colonne].mean()) / df_copy[colonne].std()
            df_copy['aberrante'] = abs(z_scores) > seuil
        
        # Traiter les valeurs aberrantes selon la méthode choisie
        if traitement == 'remplacer':
            # Remplacer les valeurs aberrantes par la moyenne ou la médiane
            median_value = df_copy[colonne].median()
            df_copy.loc[df_copy['aberrante'], colonne] = median_value
            logger.info(f"Valeurs aberrantes remplacées par la médiane ({median_value})")
        
        elif traitement == 'supprimer':
            # Supprimer les lignes contenant des valeurs aberrantes
            n_before = len(df_copy)
            df_copy = df_copy[~df_copy['aberrante']]
            n_after = len(df_copy)
            logger.info(f"{n_before - n_after} lignes contenant des valeurs aberrantes supprimées")
        
        # Pour 'marquer', on ne fait rien de plus que créer la colonne 'aberrante'
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erreur lors de la détection des valeurs aberrantes: {e}")
        return df

def tester_stationnarite(serie: pd.Series, nom_serie: str) -> Tuple[bool, float, Dict[str, float]]:
    """
    Teste la stationnarité d'une série temporelle avec le test ADF.
    
    Args:
        serie: Série temporelle à tester
        nom_serie: Nom de la série pour l'affichage
        
    Returns:
        Tuple contenant (est_stationnaire, p_value, résultats du test)
    """
    try:
        # Supprimer les valeurs manquantes
        serie = serie.dropna()
        
        if len(serie) < 20:
            logger.warning(f"Série {nom_serie} trop courte pour le test de stationnarité ({len(serie)} observations)")
            return False, 1.0, {}
        
        # Test ADF
        result = adfuller(serie)
        
        # Extraire les résultats
        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        # Déterminer si la série est stationnaire
        est_stationnaire = p_value < 0.05
        
        # Afficher les résultats
        logger.info(f"Test de stationnarité pour {nom_serie}:")
        logger.info(f"  Statistique ADF: {adf_stat:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Valeurs critiques:")
        for key, value in critical_values.items():
            logger.info(f"    {key}: {value:.4f}")
        logger.info(f"  Conclusion: {'Stationnaire' if est_stationnaire else 'Non stationnaire'}")
        
        # Retourner les résultats
        resultats = {
            'adf_stat': adf_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'est_stationnaire': est_stationnaire
        }
        
        return est_stationnaire, p_value, resultats
    
    except Exception as e:
        logger.error(f"Erreur lors du test de stationnarité pour {nom_serie}: {e}")
        return False, 1.0, {}

def differencier_serie(serie: pd.Series, ordre: int = 1) -> pd.Series:
    """
    Différencie une série temporelle pour la rendre stationnaire.
    
    Args:
        serie: Série temporelle à différencier
        ordre: Ordre de différenciation
        
    Returns:
        Série différenciée
    """
    try:
        serie_diff = serie.copy()
        
        for i in range(ordre):
            serie_diff = serie_diff.diff()
        
        # Supprimer les valeurs manquantes
        serie_diff = serie_diff.dropna()
        
        return serie_diff
    
    except Exception as e:
        logger.error(f"Erreur lors de la différenciation de la série: {e}")
        return serie

def generer_statistiques_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère des statistiques descriptives pour un DataFrame.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        DataFrame contenant les statistiques descriptives
    """
    try:
        # Sélectionner uniquement les colonnes numériques
        df_num = df.select_dtypes(include=[np.number])
        
        if df_num.empty:
            logger.warning("Aucune colonne numérique trouvée pour les statistiques descriptives")
            return pd.DataFrame()
        
        # Calculer les statistiques descriptives
        stats = df_num.describe()
        
        # Ajouter des statistiques supplémentaires
        stats.loc['skew'] = df_num.skew()
        stats.loc['kurtosis'] = df_num.kurtosis()
        stats.loc['missing'] = df_num.isnull().sum()
        stats.loc['missing_pct'] = df_num.isnull().sum() / len(df_num) * 100
        
        return stats
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des statistiques descriptives: {e}")
        return pd.DataFrame()

def visualiser_serie_temporelle(df: pd.DataFrame, colonne: str, titre: str = None, figsize: Tuple[int, int] = (12, 6), 
                               sauvegarder: bool = False, chemin_sauvegarde: str = None) -> None:
    """
    Visualise une série temporelle.
    
    Args:
        df: DataFrame contenant les données
        colonne: Nom de la colonne à visualiser
        titre: Titre du graphique
        figsize: Taille de la figure
        sauvegarder: Si True, sauvegarde le graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Vérifier que la colonne existe
        if colonne not in df.columns:
            logger.warning(f"La colonne {colonne} n'existe pas dans le DataFrame.")
            return
        
        # Vérifier que l'index est un DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("L'index n'est pas un DatetimeIndex. Visualisation impossible.")
            return
        
        # Créer la figure
        plt.figure(figsize=figsize)
        
        # Tracer la série temporelle
        plt.plot(df.index, df[colonne])
        
        # Ajouter un titre
        if titre:
            plt.title(titre)
        else:
            plt.title(f"Série temporelle de {colonne}")
        
        # Ajouter des étiquettes
        plt.xlabel('Date')
        plt.ylabel(colonne)
        
        # Ajouter une grille
        plt.grid(True, alpha=0.3)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder le graphique si demandé
        if sauvegarder and chemin_sauvegarde:
            # Créer le répertoire de sortie s'il n'existe pas
            os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
            plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegardé dans {chemin_sauvegarde}")
        
        # Afficher le graphique
        plt.show()
    
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation de la série temporelle: {e}")

def tracer_matrice_correlation(df: pd.DataFrame, titre: str, chemin_sauvegarde: str) -> None:
    """
    Trace et sauvegarde une matrice de corrélation.
    
    Args:
        df: DataFrame contenant les données
        titre: Titre du graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Calculer la matrice de corrélation
        corr = df.corr()
        
        # Créer la figure
        plt.figure(figsize=(12, 8))
        
        # Tracer la matrice de corrélation
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                   fmt='.2f', square=True, linewidths=0.5, cbar_kws={'label': 'Coefficient de corrélation'})
        
        # Ajouter un titre
        plt.title(titre)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        
        # Sauvegarder le graphique
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Matrice de corrélation sauvegardée dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors du traçage de la matrice de corrélation: {e}")

def tracer_heatmap(df: pd.DataFrame, titre: str, chemin_sauvegarde: str) -> None:
    """
    Trace et sauvegarde un heatmap.
    
    Args:
        df: DataFrame contenant les données
        titre: Titre du graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Créer la figure
        plt.figure(figsize=(12, 8))
        
        # Tracer le heatmap
        sns.heatmap(df, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', linewidths=0.5)
        
        # Ajouter un titre
        plt.title(titre)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        
        # Sauvegarder le graphique
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap sauvegardé dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors du traçage du heatmap: {e}")

def tracer_volatilite(df: pd.DataFrame, colonne: str, titre: str, chemin_sauvegarde: str) -> None:
    """
    Trace et sauvegarde un graphique de volatilité.
    
    Args:
        df: DataFrame contenant les données
        colonne: Nom de la colonne de volatilité
        titre: Titre du graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Vérifier que la colonne existe
        if colonne not in df.columns:
            logger.warning(f"La colonne {colonne} n'existe pas dans le DataFrame.")
            return
        
        # Créer la figure
        plt.figure(figsize=(12, 6))
        
        # Tracer la volatilité
        plt.plot(df.index, df[colonne], linewidth=1.5)
        
        # Ajouter un titre
        plt.title(titre)
        
        # Ajouter des étiquettes
        plt.xlabel('Date')
        plt.ylabel('Volatilité')
        
        # Ajouter une grille
        plt.grid(True, alpha=0.3)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        
        # Sauvegarder le graphique
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique de volatilité sauvegardé dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors du traçage du graphique de volatilité: {e}")

def tracer_comparaison_volatilites(df: pd.DataFrame, colonnes: List[str], titre: str, chemin_sauvegarde: str) -> None:
    """
    Trace et sauvegarde un graphique comparant plusieurs volatilités.
    
    Args:
        df: DataFrame contenant les données
        colonnes: Liste des noms de colonnes de volatilité
        titre: Titre du graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Vérifier que les colonnes existent
        colonnes_existantes = [col for col in colonnes if col in df.columns]
        if not colonnes_existantes:
            logger.warning("Aucune des colonnes spécifiées n'existe dans le DataFrame.")
            return
        
        # Créer la figure
        plt.figure(figsize=(12, 6))
        
        # Tracer les volatilités
        for col in colonnes_existantes:
            plt.plot(df.index, df[col], label=col.replace('volatilite_', '').replace('_', ' ').title(), linewidth=1.5)
        
        # Ajouter un titre
        plt.title(titre)
        
        # Ajouter des étiquettes
        plt.xlabel('Date')
        plt.ylabel('Volatilité')
        
        # Ajouter une légende
        plt.legend()
        
        # Ajouter une grille
        plt.grid(True, alpha=0.3)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        
        # Sauvegarder le graphique
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique de comparaison des volatilités sauvegardé dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors du traçage du graphique de comparaison des volatilités: {e}")

def creer_tableau_bord_interactif(df: pd.DataFrame, titre: str, colonnes_volatilite: List[str], 
                                 colonne_prix: str, chemin_sauvegarde: str) -> None:
    """
    Crée et sauvegarde un tableau de bord interactif avec Plotly.
    
    Args:
        df: DataFrame contenant les données
        titre: Titre du tableau de bord
        colonnes_volatilite: Liste des noms de colonnes de volatilité
        colonne_prix: Nom de la colonne de prix
        chemin_sauvegarde: Chemin où sauvegarder le tableau de bord
    """
    try:
        # Vérifier que les colonnes existent
        colonnes_volatilite_existantes = [col for col in colonnes_volatilite if col in df.columns]
        if not colonnes_volatilite_existantes:
            logger.warning("Aucune des colonnes de volatilité spécifiées n'existe dans le DataFrame.")
            return
        
        if colonne_prix not in df.columns:
            logger.warning(f"La colonne de prix {colonne_prix} n'existe pas dans le DataFrame.")
            return
        
        # Créer la figure avec deux sous-graphiques
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           subplot_titles=('Volatilités', 'Prix'),
                           vertical_spacing=0.1, row_heights=[0.6, 0.4])
        
        # Ajouter les traces de volatilité
        for col in colonnes_volatilite_existantes:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=col.replace('volatilite_', '').replace('_', ' ').title()),
                row=1, col=1
            )
        
        # Ajouter la trace de prix
        fig.add_trace(
            go.Scatter(x=df.index, y=df[colonne_prix], name=colonne_prix.replace('_', ' ').title()),
            row=2, col=1
        )
        
        # Mettre à jour la mise en page
        fig.update_layout(
            title=titre,
            height=800,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        # Mettre à jour les axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volatilité", row=1, col=1)
        fig.update_yaxes(title_text="Prix", row=2, col=1)
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        
        # Sauvegarder le tableau de bord
        fig.write_html(chemin_sauvegarde)
        
        logger.info(f"Tableau de bord interactif sauvegardé dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la création du tableau de bord interactif: {e}")

def creer_validation_croisee_temporelle(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Crée des ensembles d'entraînement et de test pour la validation croisée temporelle.
    
    Args:
        X: DataFrame des variables explicatives
        y: Série de la variable cible
        n_splits: Nombre de divisions
        
    Returns:
        Liste de tuples (indices d'entraînement, indices de test)
    """
    try:
        # Créer l'objet TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Générer les indices d'entraînement et de test
        splits = list(tscv.split(X))
        
        logger.info(f"Validation croisée temporelle créée avec {n_splits} divisions")
        
        return splits
    
    except Exception as e:
        logger.error(f"Erreur lors de la création de la validation croisée temporelle: {e}")
        return []

def evaluer_modele(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Évalue les performances d'un modèle de prédiction.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        
    Returns:
        Dictionnaire des métriques d'évaluation
    """
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Calculer les métriques
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculer le MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
        
        # Créer le dictionnaire des métriques
        metriques = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        logger.info("Métriques d'évaluation calculées:")
        for metrique, valeur in metriques.items():
            logger.info(f"  {metrique.upper()}: {valeur:.4f}")
        
        return metriques
    
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle: {e}")
        return {}

def visualiser_predictions(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.DatetimeIndex, 
                          titre: str, chemin_sauvegarde: str) -> None:
    """
    Visualise les prédictions d'un modèle.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        dates: Index des dates
        titre: Titre du graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Créer la figure
        plt.figure(figsize=(12, 6))
        
        # Tracer les valeurs réelles et prédites
        plt.plot(dates, y_true, label='Réel', linewidth=1.5)
        plt.plot(dates, y_pred, label='Prédit', linewidth=1.5, linestyle='--')
        
        # Ajouter un titre
        plt.title(titre)
        
        # Ajouter des étiquettes
        plt.xlabel('Date')
        plt.ylabel('Valeur')
        
        # Ajouter une légende
        plt.legend()
        
        # Ajouter une grille
        plt.grid(True, alpha=0.3)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        
        # Sauvegarder le graphique
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique des prédictions sauvegardé dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation des prédictions: {e}")

def visualiser_distribution_erreurs(y_true: np.ndarray, y_pred: np.ndarray, 
                                  titre: str, chemin_sauvegarde: str) -> None:
    """
    Visualise la distribution des erreurs de prédiction.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        titre: Titre du graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Calculer les erreurs
        erreurs = y_true - y_pred
        
        # Créer la figure
        plt.figure(figsize=(12, 6))
        
        # Tracer l'histogramme des erreurs
        plt.hist(erreurs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Ajouter une ligne verticale à zéro
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        
        # Ajouter un titre
        plt.title(titre)
        
        # Ajouter des étiquettes
        plt.xlabel('Erreur')
        plt.ylabel('Fréquence')
        
        # Ajouter une grille
        plt.grid(True, alpha=0.3)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        
        # Sauvegarder le graphique
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Créer également une version interactive avec Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=erreurs,
            nbinsx=30,
            marker_color='skyblue',
            opacity=0.7
        ))
        
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=0, y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig.update_layout(
            title=titre,
            xaxis_title="Erreur",
            yaxis_title="Fréquence",
            template="plotly_white"
        )
        
        # Sauvegarder la version interactive
        chemin_interactif = chemin_sauvegarde.replace('.png', '.html')
        fig.write_html(chemin_interactif)
        
        logger.info(f"Graphique de distribution des erreurs sauvegardé dans {chemin_sauvegarde}")
        logger.info(f"Version interactive sauvegardée dans {chemin_interactif}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation de la distribution des erreurs: {e}")

def tracer_importance_features(importance_scores: Dict[str, float], titre: str, chemin_sauvegarde: str) -> None:
    """
    Trace un graphique de l'importance des features.
    
    Args:
        importance_scores: Dictionnaire des scores d'importance des features
        titre: Titre du graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Créer le répertoire de sauvegarde si nécessaire
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        
        # Trier les features par importance décroissante
        sorted_features = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        # Créer la figure
        plt.figure(figsize=(12, 6))
        
        # Tracer le graphique en barres horizontales
        plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
        
        # Personnaliser le graphique
        plt.title(titre, fontsize=14, pad=20)
        plt.xlabel('Score d\'importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique d'importance des features sauvegardé dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors du tracé de l'importance des features: {e}")
def tracer_serie_temporelle(df: pd.DataFrame, colonne: str, titre: str, chemin_sauvegarde: str) -> None:
    """
    Trace et sauvegarde une série temporelle.
    
    Args:
        df: DataFrame contenant les données
        colonne: Nom de la colonne à tracer
        titre: Titre du graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Vérifier que la colonne existe
        if colonne not in df.columns:
            logger.warning(f"La colonne {colonne} n'existe pas dans le DataFrame.")
            return
        
        # Créer la figure
        plt.figure(figsize=(12, 6))
        
        # Tracer la série temporelle
        plt.plot(df.index, df[colonne], linewidth=1.5)
        
        # Ajouter les éléments du graphique
        plt.title(titre)
        plt.xlabel('Date')
        plt.ylabel(colonne)
        plt.grid(True, alpha=0.3)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Série temporelle sauvegardée dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors du traçage de la série temporelle: {e}")

def analyser_importance_variables(model, X: pd.DataFrame, titre: str, chemin_sauvegarde: str) -> None:
    """
    Analyse et visualise l'importance des variables dans un modèle.
    
    Args:
        model: Modèle entraîné (doit avoir un attribut feature_importances_)
        X: DataFrame des variables explicatives
        titre: Titre du graphique
        chemin_sauvegarde: Chemin où sauvegarder le graphique
    """
    try:
        # Vérifier si le modèle a un attribut feature_importances_
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Le modèle ne possède pas d'attribut feature_importances_.")
            return
        
        # Extraire l'importance des variables
        importances = model.feature_importances_
        
        # Créer un DataFrame pour l'importance des variables
        importance_df = pd.DataFrame({
            'Variable': X.columns,
            'Importance': importances
        })
        
        # Trier par importance décroissante
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Créer la figure
        plt.figure(figsize=(12, 8))
        
        # Tracer l'importance des variables
        sns.barplot(x='Importance', y='Variable', data=importance_df)
        
        # Ajouter un titre
        plt.title(titre)
        
        # Ajouter des étiquettes
        plt.xlabel('Importance')
        plt.ylabel('Variable')
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        
        # Sauvegarder le graphique
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique d'importance des variables sauvegardé dans {chemin_sauvegarde}")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de l'importance des variables: {e}")
