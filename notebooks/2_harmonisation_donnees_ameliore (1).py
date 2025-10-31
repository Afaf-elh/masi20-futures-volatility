import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import warnings
from functools import reduce
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, EasterMonday, GoodFriday
from pandas.tseries.offsets import CustomBusinessDay


# S'assurer que les warnings ne polluent pas la sortie
warnings.filterwarnings("ignore")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Définition de l'amplitude temporelle et de la fréquence finale journalière
DATE_START = "2015-01-01"
DATE_END = "2024-12-31"

# --- Paramètres spécifiques au contrat MASI20 selon l'AMMC ---
MASI20_TAILLE_CONTRAT = 10  # MAD par point d'indice
MASI20_DEPOT_GARANTIE_INITIAL = 1000  # DH
MASI20_ECHEANCES_MOIS = [3, 6, 9, 12]  # Mars, Juin, Septembre, Décembre
MASI20_JOUR_ECHEANCE_SEMAINE = 4 # Vendredi (0=Lundi, 4=Vendredi)
MASI20_SEMAINE_ECHEANCE = 3 # Troisième vendredi du mois
MASI20_BASE_CALCUL_JOURS_ANNUELS = 360
MASI20_TAUX_INTERET_SANS_RISQUE_ANNUEL = 0.025 # Exemple, à ajuster ou rendre dynamique
MASI20_TAUX_DIVIDENDE_ANNUEL = 0.00 # Exemple, à ajuster ou rendre dynamique. Le document AMMC indique que le calcul en tient compte.
# --------------------------------------------------------------

# Dictionnaire de fréquence par type de fichier
FREQ_MAPPING = {
    'gdp': 'Q',
    'pib': 'Q',
    'taux': 'M',
    'inflation': 'M',
    'indice': 'D',
    'future': 'D',
    'usd': 'D',
    'eur': 'D'
}

TYPES_CLOSE_ONLY = {"indice", "future", "usd", "eur"}

##############################################
# Fonctions de calendrier pour jours ouvrés Maroc (Exemple simplifié)
##############################################
class MoroccoHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("Nouvel An", month=1, day=1),
        Holiday("Manifeste de l'Indépendance", month=1, day=11),
        Holiday("Fête du Travail", month=5, day=1),
        Holiday("Fête du Trône", month=7, day=30),
        Holiday("Allégeance Oued Eddahab", month=8, day=14),
        Holiday("Révolution du Roi et du Peuple", month=8, day=20),
        Holiday("Fête de la Jeunesse", month=8, day=21),
        Holiday("Marche Verte", month=11, day=6),
        Holiday("Fête de l'Indépendance", month=11, day=18),
    ]

morocco_calendar = CustomBusinessDay(calendar=MoroccoHolidayCalendar())

##############################################
# Fonctions de traitement des DataFrames
##############################################

def detecter_type_fichier(nom_fichier):
    lower = nom_fichier.lower()
    if "jse40_future" in lower:
        return "future"
    if "jse40" in lower:
        return "indice"
    for key in FREQ_MAPPING:
        if key in lower:
            return key
    return "autre"

def standardiser_dataframe(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    date_cols = [col for col in df.columns if "date" in col]
    if date_cols:
        formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]
        date_parsed = None
        for fmt in formats:
            try:
                date_parsed = pd.to_datetime(df[date_cols[0]], format=fmt, errors="raise")
                logger.info(f"Conversion de la colonne de date avec le format {fmt}")
                break
            except Exception:
                continue
        if date_parsed is None:
            date_parsed = pd.to_datetime(df[date_cols[0]], errors="coerce")
        df["date"] = date_parsed
        df = df.drop(columns=[c for c in date_cols if c != "date"])
    else:
        df["date"] = pd.date_range(start=DATE_START, end=DATE_END, periods=len(df))
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df

def reindexer_df(df):
    full_index = pd.date_range(start=DATE_START, end=DATE_END, freq="D")
    if "date" not in df.columns:
        df = df.reset_index()
    df = df.set_index("date")
    df = df.reindex(full_index)
    df.index.name = "date"
    df = df.ffill().bfill()
    return df.reset_index()

def filtrer_colonnes_close(df, type_fichier):
    if type_fichier in TYPES_CLOSE_ONLY:
        if "close" not in df.columns and "price" in df.columns:
            df = df.rename(columns={"price": "close"})
        if "close" in df.columns:
            suffix = {"indice": "close_indice", "future": "close_future", 
                      "usd": "close_usd", "eur": "close_euro"}.get(type_fichier, "close")
            df = df[["date", "close"]]
            df = df.rename(columns={"close": suffix})
            return df
        else:
            logger.warning(f"Aucune colonne 'close' (ou 'price') trouvée pour le type {type_fichier}")
    return df

def corriger_noms_colonnes(df, pays):
    rename_map = {
        'close_indice_indice': 'close_indice',
        'close_future_future': 'close_future',
        'close_usd_usd': 'close_usd',
        'close_euro_eur': 'close_euro',
        'taux directeur_taux': 'taux_directeur',
        'inflation_autre': 'inflation',
        'close_autre': 'close_indice',
        'close_autre_x': 'close_indice',
        'close_autre_y': 'close_indice',
        'constant prices (zar million)_gdp': 'gdp',
        'constant prices (inr billion)_gdp': 'gdp',
        'constant prices (mad million)_gdp': 'gdp',
        'constant prices (try mille)_gdp': 'gdp',
        'constant prices (vnd billion)_gdp': 'gdp',
        'rendement_autre': 'rendement'
    }
    return df.rename(columns={col: rename_map.get(col, col) for col in df.columns})

def filtrer_colonnes_finales(df, pays):
    colonnes_souhaitees = [
        'date', 'inflation', 'gdp', 'close_euro', 'close_usd',
        'taux_directeur', 'close_indice', 'close_future'
    ]
    if pays == "maroc":
        colonnes_souhaitees += [
            'rendement'
        ]
    return df[[col for col in colonnes_souhaitees if col in df.columns]]

def limiter_duree_selon_indice_future(df, nom_pays):
    df = df.dropna(subset=['date'])
    df['date'] = pd.to_datetime(df['date'])

    PLAGES_PAYS = {
        'maroc': ("2020-01-01", "2024-12-31"),
        'afrique': ("2015-01-01", "2024-12-31"),
        'india': ("2015-01-01", "2024-12-31"),
        'vietnam': ("2017-08-10", "2024-12-31")
    }

    date_min, date_max = PLAGES_PAYS.get(nom_pays.lower(), (DATE_START, DATE_END))
    date_min = pd.to_datetime(date_min)
    date_max = pd.to_datetime(date_max)

    df = df[(df['date'] >= date_min) & (df['date'] <= date_max)]
    logger.info(f"📆 Données forcées entre {date_min.date()} et {date_max.date()} pour {nom_pays.upper()}")
    return df

def trouver_prochaine_echeance_masi20(date_observation):
    date_observation = pd.to_datetime(date_observation)
    prochaine_echeance = None
    for annee_offset in range(2):
        annee_actuelle = date_observation.year + annee_offset
        for mois_echeance in MASI20_ECHEANCES_MOIS:
            premier_jour_mois_echeance = datetime(annee_actuelle, mois_echeance, 1)
            jour_semaine_premier_jour = premier_jour_mois_echeance.weekday()
            jours_a_ajouter_pour_premier_vendredi = (MASI20_JOUR_ECHEANCE_SEMAINE - jour_semaine_premier_jour + 7) % 7
            premier_vendredi = premier_jour_mois_echeance + timedelta(days=jours_a_ajouter_pour_premier_vendredi)
            troisieme_vendredi = premier_vendredi + timedelta(days=14)
            date_echeance_calculee = pd.to_datetime(troisieme_vendredi)
            # TODO: Ajouter la logique de vérification jour ouvré et repli si férié.
            if date_echeance_calculee > date_observation:
                if prochaine_echeance is None or date_echeance_calculee < prochaine_echeance:
                    prochaine_echeance = date_echeance_calculee
        if prochaine_echeance is not None and annee_offset == 0: 
            break
    return prochaine_echeance

def ajouter_colonnes_futures_journalier(df, colonne_spot='close_indice', taux_annuel=MASI20_TAUX_INTERET_SANS_RISQUE_ANNUEL - MASI20_TAUX_DIVIDENDE_ANNUEL, maturite_initiale_jours=None):
    """
    Calcule une colonne 'close_future' pour le MASI20 basée sur les conditions de l'AMMC.
    Utilise une échéance calendaire (3ème vendredi des mois spécifiés) et la formule F = S * exp(taux_net * T).
    Le paramètre 'taux_annuel' est interprété comme le taux net (taux sans risque - taux de dividende).
    Le paramètre 'maturite_initiale_jours' est ignoré pour ce calcul spécifique.
    
    Args:
        df: DataFrame contenant la colonne spot et une colonne 'date'.
        colonne_spot: nom de la colonne contenant le prix spot (ex: 'close').
        taux_annuel: taux d'intérêt net annuel (taux sans risque - taux de dividende).
        maturite_initiale_jours: (Ignoré pour ce calcul) nombre de jours jusqu'à l'échéance initiale.
        
    Returns:
        DataFrame avec la colonne 'close_future' calculée selon les règles AMMC.
    """
    logger.info(f"Début du calcul de 'close_future' selon les règles AMMC pour la colonne spot: {colonne_spot}")

    if 'date' not in df.columns:
        logger.error("La colonne 'date' est manquante dans le DataFrame. Impossible de calculer les futures.")
        df_copy = df.copy()
        df_copy['close_future'] = np.nan
        return df_copy

    if colonne_spot not in df.columns:
        logger.warning(f"La colonne spot '{colonne_spot}' est manquante. La colonne 'close_future' sera remplie de NaN.")
        df_copy = df.copy()
        df_copy['close_future'] = np.nan
        return df_copy

    if df[colonne_spot].isnull().all():
        logger.warning(f"La colonne spot '{colonne_spot}' ne contient que des NaN. La colonne 'close_future' sera remplie de NaN.")
        df_copy = df.copy()
        df_copy['close_future'] = np.nan
        return df_copy

    df_calc = df.copy()
    df_calc['date'] = pd.to_datetime(df_calc['date'])

    df_calc['prochaine_echeance'] = df_calc['date'].apply(trouver_prochaine_echeance_masi20)

    if df_calc['prochaine_echeance'].isnull().any():
        logger.warning("Certaines dates n'ont pas pu trouver de prochaine échéance MASI20. 'close_future' sera NaN pour ces dates.")
        df_calc.loc[df_calc['prochaine_echeance'].isnull(), 'prochaine_echeance'] = pd.NaT

    df_calc['jours_restants'] = (df_calc['prochaine_echeance'] - df_calc['date']).dt.days
    df_calc['T_future'] = df_calc['jours_restants'] / MASI20_BASE_CALCUL_JOURS_ANNUELS
    df_calc.loc[df_calc['T_future'] <= 0, 'T_future'] = 1 / MASI20_BASE_CALCUL_JOURS_ANNUELS

    spot_prices = df_calc[colonne_spot].values
    T_values = df_calc['T_future'].values
    future_prices = spot_prices * np.exp(taux_annuel * T_values)
    df_calc['close_future'] = future_prices

    logger.info(f"Calcul de 'close_future' (AMMC) terminé.")
    return df_calc[['date', 'close_future'] + [col for col in df.columns if col not in ['date', 'close_future']]]

def harmoniser_pays(nom_pays, dossier_path):
    logger.info(f"\n--- Traitement de {nom_pays.upper()} ---")
    fichiers = os.listdir(dossier_path)
    dataframes = []

    for fichier in fichiers:
        path = os.path.join(dossier_path, fichier)
        try:
            if fichier.endswith(".csv"):
                df = pd.read_csv(path)
            elif fichier.endswith(".xlsx"):
                df = pd.read_excel(path)
            else:
                continue

            df = standardiser_dataframe(df)
            df = df.apply(lambda x: pd.to_numeric(x, errors="ignore") if x.name != "date" else x)
            type_fichier = detecter_type_fichier(fichier)
            df = filtrer_colonnes_close(df, type_fichier)
            df = reindexer_df(df)
            df = df.rename(columns={col: f"{col}_{type_fichier}" for col in df.columns if col != "date"})
            dataframes.append(df)
        except Exception as e:
            logger.warning(f"Erreur avec {fichier}: {e}")

    if dataframes:
        df_merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), dataframes)
        df_merged = reindexer_df(df_merged)
        df_corrige = corriger_noms_colonnes(df_merged, nom_pays)
        df_corrige = limiter_duree_selon_indice_future(df_corrige, nom_pays)
        # Pour le Maroc, on calcule les contrats futures analytiques
        if nom_pays.lower() == "maroc":
            df_corrige = ajouter_colonnes_futures_journalier(df_corrige)

        df_final = filtrer_colonnes_finales(df_corrige, nom_pays)


        save_path = os.path.join("data_harmonisee", nom_pays.lower(), "donnees_fusionnees_final.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_final.to_csv(save_path, index=False)
        logger.info(f"✅ Données harmonisées finales sauvegardées pour {nom_pays.lower()}.")
    else:
        logger.error(f"Aucun fichier de données valide pour {nom_pays}.")

# La fonction generer_rapport_harmonisation() reste inchangée

def generer_rapport_harmonisation(chemins, pays_configs):
    rapport_dir = os.path.join(chemins["data_harmonisee"], "..", "rapport_final")
    os.makedirs(rapport_dir, exist_ok=True)
    rapport_path = os.path.join(rapport_dir, "rapport_harmonisation.md")

    with open(rapport_path, "w", encoding="utf-8") as f:
        f.write("# Rapport d'Harmonisation des Données\n\n")
        f.write(f"Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Résumé Global\n\n")
        f.write(f"- Nombre de pays analysés : {len(pays_configs)}\n")
        f.write(f"- Période d'analyse : {DATE_START} à {DATE_END}\n\n")

        for config in pays_configs:
            pays_nom = config["nom"]
            f.write(f"## {pays_nom.capitalize()}\n\n")
            dossier_pays = os.path.join(chemins["data_harmonisee"], config["dossier"])
            fichier_fusion = os.path.join(dossier_pays, "donnees_fusionnees.csv")
            if os.path.exists(fichier_fusion):
                df = pd.read_csv(fichier_fusion)
                date_cols = [col for col in df.columns if "date" in col.lower()]
                if date_cols:
                    try:
                        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
                        period = f"du {df[date_cols[0]].min().strftime('%Y-%m-%d')} au {df[date_cols[0]].max().strftime('%Y-%m-%d')}"
                    except Exception:
                        period = "Non disponible"
                else:
                    period = "Non disponible (colonne de date non trouvée)"
                f.write("### Statistiques\n")
                f.write(f"- Nombre de lignes : {len(df)}\n")
                f.write(f"- Nombre de colonnes : {len(df.columns)}\n")
                f.write(f"- Période : {period}\n\n")
            else:
                f.write("Aucune donnée harmonisée disponible.\n\n")

        f.write("## Méthodologie\n\n")
        f.write("1. Standardisation des noms de colonnes.\n")
        f.write("2. Conversion explicite de la date.\n")
        f.write("3. Rééchantillonnage quotidien avec ffill/bfill.\n")
        f.write("4. Conservation uniquement des colonnes pertinentes.\n")
        f.write("5. Fusion sur la colonne date.\n")
        f.write("6. Génération du rapport.\n")

#########################################
# Exécution principale du script
#########################################
if __name__ == "__main__":
    pays_configs = [
        {"nom": "maroc", "dossier": "Maroc"},
        {"nom": "vietnam", "dossier": "Vietnam"},
        {"nom": "afrique", "dossier": "Afrique"},
        {"nom": "india", "dossier": "India"}
    ]

    for config in pays_configs:
        dossier_source = os.path.join("data", config["dossier"])
        harmoniser_pays(config["nom"], dossier_source)

    generer_rapport_harmonisation({"data_harmonisee": "data_harmonisee"}, pays_configs)
    logger.info("Harmonisation des données terminée.")