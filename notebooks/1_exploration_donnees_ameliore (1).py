"""
Script d'exploration des données pour l'analyse de la volatilité des futures sur indices
boursiers pour plusieurs pays (Maroc, Vietnam, Afrique du Sud, Inde).
Version améliorée avec intégration des modules utils et config.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
from statsmodels.tsa.stattools import adfuller

# Importer les modules utilitaires et de configuration
from utils import (
    charger_donnees, sauvegarder_donnees, generer_statistiques_descriptives,
    detecter_valeurs_aberrantes, installer_dependances
)
from config import CHEMINS, PAYS

# S'assurer que toutes les dépendances sont installées
installer_dependances()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

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

def explorer_fichier(chemin_fichier: str) -> Dict[str, Any]:
    """
    Explore un fichier de données et génère des statistiques descriptives.
    
    Args:
        chemin_fichier: Chemin du fichier à explorer
        
    Returns:
        Dictionnaire contenant les résultats de l'exploration
    """
    try:
        # Charger les données
        df = charger_donnees(chemin_fichier)
        
        if df is None or df.empty:
            logger.error(f"Impossible de charger le fichier {chemin_fichier}")
            return {
                'fichier': os.path.basename(chemin_fichier),
                'statut': 'erreur',
                'message': 'Impossible de charger le fichier'
            }
        
        # Informations générales
        nb_lignes, nb_colonnes = df.shape
        
        # Statistiques descriptives
        stats = generer_statistiques_descriptives(df)
        
        # Valeurs manquantes
        valeurs_manquantes = df.isnull().sum()
        pourcentage_manquant = valeurs_manquantes / nb_lignes * 100
        
        # Colonnes de date
        colonnes_date = []
        for col in df.columns:
            try:
                # Convertir temporairement en datetime pour l'analyse
                temp_dates = pd.to_datetime(df[col], errors='coerce')
                if not temp_dates.isna().all():
                    min_date = temp_dates.min()
                    max_date = temp_dates.max()
                    unique_dates = temp_dates.nunique()
                    colonnes_date.append({
                        'colonne': col,
                        'min_date': min_date,
                        'max_date': max_date,
                        'nb_dates_uniques': unique_dates
                    })
            except Exception as e:
                logger.debug(f"Erreur lors de l'analyse des dates pour la colonne {col}: {e}")
        
        # Valeurs aberrantes
        valeurs_aberrantes = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                df_aberrantes = detecter_valeurs_aberrantes(df, col, methode='zscore', seuil=3.0)
                nb_aberrantes = df_aberrantes['aberrante'].sum()
                if nb_aberrantes > 0:
                    valeurs_aberrantes[col] = {
                        'nombre': nb_aberrantes,
                        'pourcentage': nb_aberrantes / nb_lignes * 100
                    }
            except Exception as e:
                logger.debug(f"Erreur lors de la détection des valeurs aberrantes pour la colonne {col}: {e}")
        
        # Tests de stationnarité pour les colonnes numériques
        stationnarite = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                est_stationnaire, p_value, resultats = tester_stationnarite(df[col], col)
                stationnarite[col] = {
                    'est_stationnaire': est_stationnaire,
                    'p_value': p_value
                }
            except Exception as e:
                logger.debug(f"Erreur lors du test de stationnarité pour la colonne {col}: {e}")
        
        # Résultats
        resultats = {
            'fichier': os.path.basename(chemin_fichier),
            'statut': 'succes',
            'nb_lignes': nb_lignes,
            'nb_colonnes': nb_colonnes,
            'colonnes': list(df.columns),
            'types_donnees': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'valeurs_manquantes': {col: {'nombre': int(val), 'pourcentage': float(pourcentage_manquant[col])} for col, val in valeurs_manquantes.items() if val > 0},
            'colonnes_date': colonnes_date,
            'valeurs_aberrantes': valeurs_aberrantes,
            'stationnarite': stationnarite
        }
        
        return resultats
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exploration du fichier {chemin_fichier}: {e}")
        return {
            'fichier': os.path.basename(chemin_fichier),
            'statut': 'erreur',
            'message': str(e)
        }

def explorer_donnees_pays(pays: str) -> Dict[str, Any]:
    """
    Explore les données pour un pays donné.
    
    Args:
        pays: Clé du pays dans le dictionnaire PAYS
        
    Returns:
        Dictionnaire contenant les résultats de l'exploration
    """
    try:
        logger.info("\n" + "=" * 80)
        logger.info(f"Exploration des données pour {PAYS[pays]['nom']}")
        logger.info("=" * 80)
        
        # Récupérer les fichiers pour ce pays
        dossier_pays = PAYS[pays]['dossier']
        chemin_dossier = os.path.join(CHEMINS['data'], dossier_pays)
        
        if not os.path.exists(chemin_dossier):
            logger.error(f"Le dossier {chemin_dossier} n'existe pas")
            return {
                'pays': PAYS[pays]['nom'],
                'statut': 'erreur',
                'message': f"Le dossier {chemin_dossier} n'existe pas"
            }
        
        # Lister tous les fichiers dans le dossier
        fichiers = os.listdir(chemin_dossier)
        
        # Explorer chaque fichier
        resultats_fichiers = []
        for fichier in fichiers:
            chemin_fichier = os.path.join(chemin_dossier, fichier)
            if os.path.isfile(chemin_fichier):
                logger.info(f"\nExploration du fichier: {fichier}")
                resultats = explorer_fichier(chemin_fichier)
                resultats_fichiers.append(resultats)
                
                # Afficher quelques informations
                if resultats['statut'] == 'succes':
                    logger.info(f"  Nombre de lignes: {resultats['nb_lignes']}")
                    logger.info(f"  Nombre de colonnes: {resultats['nb_colonnes']}")
                    
                    # Afficher les colonnes de date
                    for col_date in resultats.get('colonnes_date', []):
                        logger.info(f"  Colonne de date: {col_date['colonne']}")
                        logger.info(f"    Plage de dates: {col_date['min_date']} à {col_date['max_date']}")
                        logger.info(f"    Nombre de dates uniques: {col_date['nb_dates_uniques']}")
                    
                    # Afficher les valeurs manquantes
                    for col, info in resultats.get('valeurs_manquantes', {}).items():
                        logger.info(f"  Valeurs manquantes dans {col}: {info['nombre']} ({info['pourcentage']:.2f}%)")
                    
                    # Afficher les valeurs aberrantes
                    for col, info in resultats.get('valeurs_aberrantes', {}).items():
                        logger.info(f"  Valeurs aberrantes dans {col}: {info['nombre']} ({info['pourcentage']:.2f}%)")
                        
                    # Afficher les résultats de stationnarité
                    for col, info in resultats.get('stationnarite', {}).items():
                        logger.info(f"  Stationnarité de {col}: {'Stationnaire' if info['est_stationnaire'] else 'Non stationnaire'} (p-value: {info['p_value']:.4f})")
                else:
                    logger.error(f"  Erreur: {resultats.get('message', 'Erreur inconnue')}")
        
        # Résultats
        resultats_pays = {
            'pays': PAYS[pays]['nom'],
            'statut': 'succes',
            'nb_fichiers': len(resultats_fichiers),
            'fichiers': resultats_fichiers
        }
        
        return resultats_pays
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exploration des données pour {PAYS[pays]['nom']}: {e}")
        return {
            'pays': PAYS[pays]['nom'],
            'statut': 'erreur',
            'message': str(e)
        }

def generer_rapport_exploration(resultats_pays: Dict[str, Dict[str, Any]]) -> None:
    """
    Génère un rapport d'exploration des données.
    
    Args:
        resultats_pays: Dictionnaire des résultats par pays
    """
    logger.info("\nGénération du rapport d'exploration des données...")
    
    # Créer le répertoire pour le rapport
    os.makedirs(os.path.join(CHEMINS['rapport_final']), exist_ok=True)
    
    # Ouvrir le fichier de rapport avec encodage UTF-8 explicite
    with open(os.path.join(CHEMINS['rapport_final'], "rapport_exploration.md"), 'w', encoding='utf-8') as f:
        f.write("# Rapport d'exploration des données\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Résumé global
        f.write("## Résumé global\n\n")
        
        total_pays = len(resultats_pays)
        total_fichiers = sum(resultats['nb_fichiers'] for resultats in resultats_pays.values() if resultats['statut'] == 'succes')
        
        f.write(f"- Nombre de pays analysés: {total_pays}\n")
        f.write(f"- Nombre total de fichiers: {total_fichiers}\n\n")
        
        # Détails par pays
        f.write("## Détails par pays\n\n")
        
        for pays, resultats in resultats_pays.items():
            f.write(f"### {resultats['pays']}\n\n")
            
            if resultats['statut'] == 'succes':
                f.write(f"- Nombre de fichiers: {resultats['nb_fichiers']}\n\n")
                
                # Tableau des fichiers
                f.write("| Fichier | Lignes | Colonnes | Plage de dates | Valeurs manquantes | Valeurs aberrantes | Stationnarité |\n")
                f.write("|---------|--------|----------|----------------|-------------------|-------------------|---------------|\n")
                
                for fichier in resultats['fichiers']:
                    if fichier['statut'] == 'succes':
                        # Plage de dates
                        plage_dates = "N/A"
                        if fichier.get('colonnes_date'):
                            min_date = min(col['min_date'] for col in fichier['colonnes_date'])
                            max_date = max(col['max_date'] for col in fichier['colonnes_date'])
                            plage_dates = f"{min_date.strftime('%Y-%m-%d')} à {max_date.strftime('%Y-%m-%d')}"
                        
                        # Valeurs manquantes
                        valeurs_manquantes = "Aucune"
                        if fichier.get('valeurs_manquantes'):
                            total_manquantes = sum(info['nombre'] for info in fichier['valeurs_manquantes'].values())
                            pourcentage = total_manquantes / (fichier['nb_lignes'] * fichier['nb_colonnes']) * 100
                            valeurs_manquantes = f"{total_manquantes} ({pourcentage:.2f}%)"
                        
                        # Valeurs aberrantes
                        valeurs_aberrantes = "Aucune"
                        if fichier.get('valeurs_aberrantes'):
                            total_aberrantes = sum(info['nombre'] for info in fichier['valeurs_aberrantes'].values())
                            pourcentage = total_aberrantes / fichier['nb_lignes'] * 100
                            valeurs_aberrantes = f"{total_aberrantes} ({pourcentage:.2f}%)"
                        
                        # Stationnarité
                        stationnarite = "N/A"
                        if fichier.get('stationnarite'):
                            stationnaires = sum(1 for info in fichier['stationnarite'].values() if info['est_stationnaire'])
                            total = len(fichier['stationnarite'])
                            if total > 0:
                                stationnarite = f"{stationnaires}/{total} colonnes"
                        
                        f.write(f"| {fichier['fichier']} | {fichier['nb_lignes']} | {fichier['nb_colonnes']} | {plage_dates} | {valeurs_manquantes} | {valeurs_aberrantes} | {stationnarite} |\n")
                    else:
                        f.write(f"| {fichier['fichier']} | Erreur | {fichier.get('message', 'Erreur inconnue')} | | | | |\n")
                
                f.write("\n")
            else:
                f.write(f"Erreur: {resultats.get('message', 'Erreur inconnue')}\n\n")
        
        # Problèmes identifiés
        f.write("## Problèmes identifiés\n\n")
        
        for pays, resultats in resultats_pays.items():
            if resultats['statut'] == 'succes':
                problemes = []
                
                for fichier in resultats['fichiers']:
                    if fichier['statut'] == 'succes':
                        # Valeurs manquantes importantes
                        for col, info in fichier.get('valeurs_manquantes', {}).items():
                            if info['pourcentage'] > 10:
                                problemes.append(f"- **{fichier['fichier']}**: {info['pourcentage']:.2f}% de valeurs manquantes dans la colonne '{col}'")
                        
                        # Valeurs aberrantes importantes
                        for col, info in fichier.get('valeurs_aberrantes', {}).items():
                            if info['pourcentage'] > 5:
                                problemes.append(f"- **{fichier['fichier']}**: {info['pourcentage']:.2f}% de valeurs aberrantes dans la colonne '{col}'")
                
                if problemes:
                    f.write(f"### {resultats['pays']}\n\n")
                    for probleme in problemes:
                        f.write(f"{probleme}\n")
                    f.write("\n")
        
        # Recommandations
        f.write("## Recommandations\n\n")
        f.write("1. **Traitement des valeurs manquantes**: Utiliser des méthodes d'interpolation pour combler les valeurs manquantes dans les séries temporelles.\n")
        f.write("2. **Harmonisation des formats de date**: Standardiser tous les formats de date au format AAAA-MM-JJ.\n")
        f.write("3. **Traitement des valeurs aberrantes**: Examiner et éventuellement corriger les valeurs aberrantes identifiées.\n")
        f.write("4. **Alignement des périodes**: S'assurer que toutes les séries temporelles couvrent la même période pour faciliter l'analyse comparative.\n")
        f.write("5. **Conversion à la fréquence journalière**: Convertir toutes les données à une fréquence journalière pour l'analyse de la volatilité.\n")
    
    logger.info(f"Rapport d'exploration sauvegardé dans '{CHEMINS['rapport_final']}/rapport_exploration.md'")

def main():
    """
    Fonction principale.
    """
    logger.info("Démarrage de l'exploration des données...")
    
    # Explorer les données pour chaque pays
    resultats_pays = {}
    
    for pays in PAYS.keys():
        try:
            resultats = explorer_donnees_pays(pays)
            resultats_pays[pays] = resultats
        except Exception as e:
            logger.error(f"Erreur lors de l'exploration pour {PAYS[pays]['nom']}: {e}")
            resultats_pays[pays] = {
                'pays': PAYS[pays]['nom'],
                'statut': 'erreur',
                'message': str(e)
            }
    
    # Générer le rapport d'exploration
    generer_rapport_exploration(resultats_pays)
    
    logger.info("Exploration des données terminée.")

if __name__ == "__main__":
    main()
