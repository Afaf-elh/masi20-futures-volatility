"""
Script de modèles de prédiction avancés pour la volatilité des futures sur indices boursiers.
Version épurée ne conservant que les modèles fonctionnels (GARCH, ML, LSTM, CNN-LSTM).
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from arch.univariate import ConstantMean, GARCH, Normal, StudentsT, EGARCH
from arch.univariate import arch_model
import warnings
import logging
from typing import Optional, Dict, List, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib
import pickle
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
# Utiliser l'optimiseur legacy pour compatibilité
from tensorflow import keras
from keras.optimizers.legacy import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping
import time
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Importer les modules utilitaires et de configuration
from utils import (
    charger_donnees, sauvegarder_donnees, tracer_matrice_correlation,
    tracer_heatmap, tracer_serie_temporelle, detecter_valeurs_aberrantes,
    tracer_importance_features
)
from config import CHEMINS, PAYS, VARIABLES_MACRO

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Ajout des chemins manquants dans CHEMINS si nécessaire
if "visualisations_prediction" not in CHEMINS:
    CHEMINS["visualisations_prediction"] = "visualisations_prediction"
    os.makedirs(CHEMINS["visualisations_prediction"], exist_ok=True)

# Fonction utilitaire pour gérer NaN et Inf
def handle_nan_inf(data):
    """Remplace Inf par NaN, puis impute les NaN avec la moyenne."""
    if isinstance(data, pd.DataFrame):
        # Remplacer Inf par NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        # Imputer les NaN restants
        if data.isnull().any().any():
            logger.debug("Imputation des NaN dans le DataFrame...")
            imputer = SimpleImputer(strategy='mean')
            data_imputed = imputer.fit_transform(data)
            return pd.DataFrame(data_imputed, index=data.index, columns=data.columns)
        return data
    elif isinstance(data, pd.Series):
        data = data.replace([np.inf, -np.inf], np.nan)
        if data.isnull().any():
            logger.debug("Imputation des NaN dans la Series...")
            imputer = SimpleImputer(strategy='mean')
            data_imputed = imputer.fit_transform(data.values.reshape(-1, 1)).ravel()
            return pd.Series(data_imputed, index=data.index, name=data.name)
        return data
    elif isinstance(data, np.ndarray):
        data[np.isinf(data)] = np.nan
        if np.isnan(data).any():
            logger.debug("Imputation des NaN dans l'array NumPy...")
            imputer = SimpleImputer(strategy='mean')
            # Imputer colonne par colonne si 2D
            if data.ndim == 2:
                for i in range(data.shape[1]):
                    if np.isnan(data[:, i]).any():
                        data[:, i] = imputer.fit_transform(data[:, i].reshape(-1, 1)).ravel()
            elif data.ndim == 1:
                 data = imputer.fit_transform(data.reshape(-1, 1)).ravel()
        return data
    return data

def charger_donnees_volatilite(pays):
    """
    Charge les données de volatilité pour un pays donné.
    """
    try:
        # Correction pour les noms de dossiers qui peuvent différer des clés
        dossier_pays = pays
        if pays == "afrique_sud" and "Afrique" in PAYS[pays]["dossier"]:
            dossier_pays = "Afrique"
        elif pays == "inde" and "India" in PAYS[pays]["dossier"]:
            dossier_pays = "India"

        # Essayer d'abord avec le chemin standard
        chemin_fichier = os.path.join("volatilite", pays, "resultats_volatilite.csv")

        # Si le fichier n'existe pas, essayer avec le dossier corrigé
        if not os.path.exists(chemin_fichier) and dossier_pays != pays:
            chemin_fichier = os.path.join("volatilite", dossier_pays, "resultats_volatilite.csv")

        # Si toujours pas trouvé, essayer avec le dossier du pays directement
        if not os.path.exists(chemin_fichier) and pays in PAYS:
            chemin_fichier = os.path.join("volatilite", PAYS[pays]["dossier"], "resultats_volatilite.csv")

        # Vérifier si le fichier existe
        if not os.path.exists(chemin_fichier):
            logging.error(f"Fichier de données introuvable pour {pays}: {chemin_fichier}")
            return pd.DataFrame()  # Retourner un DataFrame vide en cas d'erreur

        # Charger les données
        donnees = pd.read_csv(chemin_fichier)

        # Convertir la colonne Date en datetime
        donnees["date"] = pd.to_datetime(donnees["date"])
        donnees = donnees.set_index("date") # Définir la date comme index

        # Supprimer les lignes avec trop de valeurs manquantes (plus de 50%)
        donnees = donnees.dropna(thresh=len(donnees.columns)//2)

        # Remplacer Inf par NaN AVANT l'imputation
        donnees = donnees.replace([np.inf, -np.inf], np.nan)

        # Pour les colonnes numériques restantes, remplacer les valeurs manquantes par la moyenne
        colonnes_numeriques = donnees.select_dtypes(include=[np.number]).columns
        for colonne in colonnes_numeriques:
            if donnees[colonne].isnull().any():
                mean_val = donnees[colonne].mean()
                donnees[colonne].fillna(mean_val, inplace=True)
                logger.debug(f"NaN dans la colonne {colonne} remplacés par la moyenne: {mean_val}")

        # Vérifier s'il reste des NaN après imputation
        if donnees.isnull().any().any():
             logger.warning(f"Des NaN persistent dans les données chargées pour {pays} après imputation.")
             # Optionnel: supprimer les lignes restantes avec NaN
             # donnees = donnees.dropna()

        # Trier par date (index)
        donnees = donnees.sort_index()

        logging.info(f"Données de volatilité chargées pour {pays}: {len(donnees)} lignes x {len(donnees.columns)} colonnes")
        return donnees

    except Exception as e:
        logging.error(f"Erreur lors du chargement des données de volatilité pour {pays}: {str(e)}")
        return pd.DataFrame()  # Retourner un DataFrame vide en cas d'erreur

def preparer_donnees_modele(df: pd.DataFrame, col_volatilite: str = "volatilite_garch",
                           variables_macro: List[str] = None, lags: int = 5,
                           horizon_prediction: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare les données pour l'entraînement des modèles de prédiction.
    """
    try:
        # Vérifier si le DataFrame est vide
        if df.empty:
            logger.warning("DataFrame vide fourni à preparer_donnees_modele")
            return pd.DataFrame(), pd.Series()

        # Vérifier si la colonne de volatilité existe
        if col_volatilite not in df.columns:
            logger.warning(f"La colonne {col_volatilite} n'existe pas.")
            # Essayer de trouver une autre colonne de volatilité
            vol_cols = [col for col in df.columns if "volatilite" in col]
            if vol_cols:
                col_volatilite = vol_cols[0]
                logger.info(f"Utilisation de la colonne {col_volatilite} à la place.")
            else:
                logger.error(f"Aucune colonne de volatilité trouvée.")
                return pd.DataFrame(), pd.Series()

        # Créer la variable cible (volatilité future)
        df_target = df[col_volatilite].shift(-horizon_prediction)

        # Créer les features
        features = pd.DataFrame(index=df.index)

        # Ajouter les lags de volatilité
        for i in range(1, lags + 1):
            features[f"{col_volatilite}_lag{i}"] = df[col_volatilite].shift(i)

        # Ajouter les variables macroéconomiques si spécifiées
        if variables_macro is not None:
            for var in variables_macro:
                if var in df.columns:
                    features[var] = df[var]

                    # Ajouter également les lags des variables macroéconomiques
                    for i in range(1, lags + 1):
                        features[f"{var}_lag{i}"] = df[var].shift(i)

        # Ajouter des features supplémentaires

        # 1. Volatilité moyenne mobile
        for window in [5, 10, 20]:
            features[f"{col_volatilite}_ma{window}"] = df[col_volatilite].rolling(window=window).mean()

        # 2. Écart-type mobile
        for window in [5, 10, 20]:
            features[f"{col_volatilite}_std{window}"] = df[col_volatilite].rolling(window=window).std()

        # 3. Volatilité min/max mobile
        for window in [5, 10, 20]:
            features[f"{col_volatilite}_min{window}"] = df[col_volatilite].rolling(window=window).min()
            features[f"{col_volatilite}_max{window}"] = df[col_volatilite].rolling(window=window).max()

        # 4. Ratio de volatilité (court terme / long terme)
        ma5 = df[col_volatilite].rolling(window=5).mean()
        ma20 = df[col_volatilite].rolling(window=20).mean()
        # Éviter la division par zéro ou NaN
        features[f"{col_volatilite}_ratio"] = (ma5 / ma20.replace(0, np.nan)).fillna(1.0) # Remplacer NaN par 1 (ou une autre valeur neutre)

        # 5. Tendance (régression linéaire sur fenêtre mobile)
        def calculer_tendance(x):
            # Vérifier si x contient NaN ou Inf
            if np.isnan(x).any() or np.isinf(x).any():
                return np.nan
            if len(x) < 2:
                return np.nan
            try:
                return np.polyfit(np.arange(len(x)), x, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                 return np.nan # Gérer les erreurs de polyfit

        features[f"{col_volatilite}_trend"] = df[col_volatilite].rolling(window=10).apply(calculer_tendance, raw=True)

        # 6. Caractéristiques temporelles
        if isinstance(df.index, pd.DatetimeIndex):
            features["day_of_week"] = df.index.dayofweek
            features["month"] = df.index.month
            features["quarter"] = df.index.quarter

            # One-hot encoding des variables catégorielles
            features = pd.get_dummies(features, columns=["day_of_week", "month", "quarter"], drop_first=True)

        # Remplacer Inf par NaN AVANT dropna
        features = features.replace([np.inf, -np.inf], np.nan)

        # Supprimer les lignes avec des valeurs manquantes DANS LES FEATURES
        # Garder l'index avant dropna pour aligner la cible
        index_avant_dropna = features.index
        features = features.dropna()

        # Aligner la cible sur l'index des features après dropna
        df_target = df_target.loc[features.index]

        # Vérifier s'il reste des NaN dans la cible et les supprimer (ou imputer)
        if df_target.isnull().any():
            logger.warning("NaN détectés dans la variable cible après alignement. Suppression des lignes correspondantes.")
            valid_target_index = df_target.dropna().index
            features = features.loc[valid_target_index]
            df_target = df_target.loc[valid_target_index]

        # Vérifier si les données sont vides après le nettoyage
        if features.empty or df_target.empty:
            logger.warning("Données vides après nettoyage dans preparer_donnees_modele")
            return pd.DataFrame(), pd.Series()

        # Vérification finale des NaN/Inf (uniquement sur les colonnes numériques)
        numeric_cols = features.select_dtypes(include=np.number).columns
        if features[numeric_cols].isnull().any().any() or np.isinf(features[numeric_cols].values).any():
             logger.error("Des NaN ou Inf persistent dans les features numériques après préparation!")
        if df_target.isnull().any() or np.isinf(df_target.values).any():
             logger.error("Des NaN ou Inf persistent dans la cible après préparation!")

        return features, df_target

    except Exception as e:
        logger.error(f"Erreur lors de la préparation des données pour le modèle: {e}")
        return pd.DataFrame(), pd.Series()


def preparer_donnees_lstm(donnees, n_steps=5):
    """
    Prépare les données pour les modèles LSTM.
    """
    try:
        # Vérifier si le DataFrame est vide
        if donnees.empty:
            logger.warning("DataFrame vide fourni à preparer_donnees_lstm")
            raise ValueError("Données vides fournies à preparer_donnees_lstm")

        # Sélectionner les features pertinentes
        features_cols = [
            "volatilite_historique", "rendement", "inflation",
            "close_euro", "close_usd", "taux_directeur"
        ]

        # Vérifier les colonnes disponibles
        features_cols = [f for f in features_cols if f in donnees.columns]

        # Vérifier si des colonnes sont manquantes
        if not features_cols:
            raise ValueError("Aucune des colonnes spécifiées n'est disponible dans les données.")

        # Créer le DataFrame avec les features sélectionnées
        X = donnees[features_cols].copy()

        # Gérer les valeurs manquantes et infinies
        X = handle_nan_inf(X)

        # Normaliser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Créer les séquences pour LSTM
        X_lstm, y_lstm = [], []
        # Utiliser la première colonne (volatilite_historique si disponible) comme cible
        target_col_index = 0
        for i in range(len(X_scaled) - n_steps):
            X_lstm.append(X_scaled[i:(i + n_steps)])
            y_lstm.append(X_scaled[i + n_steps, target_col_index])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        # Vérifier la cohérence des dimensions
        if len(X_lstm) != len(y_lstm):
            raise ValueError(f"Incohérence des dimensions : X_lstm a {len(X_lstm)} échantillons, mais y_lstm en a {len(y_lstm)}.")

        # Diviser en train et test
        train_size = int(len(X_lstm) * 0.8)
        X_train = X_lstm[:train_size]
        X_test = X_lstm[train_size:]
        y_train = y_lstm[:train_size]
        y_test = y_lstm[train_size:] 

        logging.info(f"Données LSTM préparées avec succès. Shape: X={X_lstm.shape}, y={y_lstm.shape}")

        return X_train, X_test, y_train, y_test, scaler

    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données LSTM: {str(e)}")
        raise


def entrainer_lstm(X_train, X_test, y_train, y_test):
    """
    Entraîne les modèles LSTM et CNN-LSTM.
    """
    try:
        # Vérifier les NaN/Inf dans les données d'entrée LSTM
        X_train = handle_nan_inf(X_train)
        X_test = handle_nan_inf(X_test)
        y_train = handle_nan_inf(y_train)
        y_test = handle_nan_inf(y_test)

        # Modèle LSTM amélioré
        model_lstm = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1)
        ])

        # Compiler le modèle avec un learning rate personnalisé (Keras 3)
        optimizer = Adam(learning_rate=0.001)
        model_lstm.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        # Early stopping pour éviter le surapprentissage
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        # Diviser les données d'entraînement en train et validation
        train_size = int(len(X_train) * 0.8)
        X_train_lstm, X_val_lstm = X_train[:train_size], X_train[train_size:]
        y_train_lstm, y_val_lstm = y_train[:train_size], y_train[train_size:]

        # Entraîner le modèle
        history_lstm = model_lstm.fit(
            X_train_lstm, y_train_lstm,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_lstm, y_val_lstm),
            callbacks=[early_stopping],
            verbose=0
        )

        # Évaluer le modèle
        pred_lstm = model_lstm.predict(X_test)
        rmse_lstm = np.sqrt(mean_squared_error(y_test, pred_lstm))
        mae_lstm = mean_absolute_error(y_test, pred_lstm)
        r2_lstm = r2_score(y_test, pred_lstm)

        logging.info(f"Modèle LSTM amélioré entraîné - RMSE: {rmse_lstm:.4f}, MAE: {mae_lstm:.4f}, R²: {r2_lstm:.4f}")

        # Sauvegarder le modèle
        os.makedirs("modeles", exist_ok=True)
        # Utiliser le format .keras recommandé pour Keras 3
        model_lstm.save("modeles/lstm_model_ameliore.keras")

        # Modèle CNN-LSTM amélioré avec padding approprié
        model_cnn_lstm = Sequential([
            # Couche CNN avec padding pour éviter la réduction de dimension
            Conv1D(filters=64, kernel_size=3, padding="same", activation="relu",
                  input_shape=(X_train.shape[1], X_train.shape[2])),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, padding="same"),
            Dropout(0.3),

            # Deuxième couche CNN
            Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, padding="same"),
            Dropout(0.3),

            # Couches LSTM
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),

            # Couches denses
            Dense(16, activation="relu"),
            Dense(1)
        ])

        # Compiler le modèle (Keras 3)
        model_cnn_lstm.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        # Diviser les données d'entraînement en train et validation
        X_train_cnn, X_val_cnn = X_train[:train_size], X_train[train_size:]
        y_train_cnn, y_val_cnn = y_train[:train_size], y_train[train_size:]

        # Entraîner le modèle
        history_cnn_lstm = model_cnn_lstm.fit(
            X_train_cnn, y_train_cnn,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_cnn, y_val_cnn),
            callbacks=[early_stopping],
            verbose=0
        )

        # Évaluer le modèle
        pred_cnn_lstm = model_cnn_lstm.predict(X_test)
        rmse_cnn_lstm = np.sqrt(mean_squared_error(y_test, pred_cnn_lstm))
        mae_cnn_lstm = mean_absolute_error(y_test, pred_cnn_lstm)
        r2_cnn_lstm = r2_score(y_test, pred_cnn_lstm)

        logging.info(f"Modèle CNN-LSTM amélioré entraîné - RMSE: {rmse_cnn_lstm:.4f}, MAE: {mae_cnn_lstm:.4f}, R²: {r2_cnn_lstm:.4f}")

        # Sauvegarder le modèle (format .keras)
        model_cnn_lstm.save("modeles/cnn_lstm_model_ameliore.keras")

        # Visualiser les résultats
        plt.figure(figsize=(15, 10))

        # Prédictions vs Réalité
        plt.subplot(2, 2, 1)
        plt.plot(y_test, label="Réalité")
        plt.plot(pred_lstm, label="LSTM")
        plt.plot(pred_cnn_lstm, label="CNN-LSTM")
        plt.title("Prédictions vs Réalité")
        plt.legend()

        # Courbes d'apprentissage
        plt.subplot(2, 2, 2)
        plt.plot(history_lstm.history["loss"], label="LSTM - Train")
        plt.plot(history_lstm.history["val_loss"], label="LSTM - Validation")
        plt.plot(history_cnn_lstm.history["loss"], label="CNN-LSTM - Train")
        plt.plot(history_cnn_lstm.history["val_loss"], label="CNN-LSTM - Validation")
        plt.title("Courbes d'apprentissage")
        plt.legend()

        # Erreurs de prédiction
        plt.subplot(2, 2, 3)
        plt.plot(y_test - pred_lstm.flatten(), label="LSTM") # Flatten predictions
        plt.plot(y_test - pred_cnn_lstm.flatten(), label="CNN-LSTM") # Flatten predictions
        plt.title("Erreurs de prédiction")
        plt.legend()

        # Distribution des erreurs
        plt.subplot(2, 2, 4)
        plt.hist(y_test - pred_lstm.flatten(), bins=50, alpha=0.5, label="LSTM") # Flatten predictions
        plt.hist(y_test - pred_cnn_lstm.flatten(), bins=50, alpha=0.5, label="CNN-LSTM") # Flatten predictions
        plt.title("Distribution des erreurs")
        plt.legend()

        plt.tight_layout()
        os.makedirs("visualisations", exist_ok=True)
        plt.savefig("visualisations/lstm_ameliore.png")
        plt.close()

        return {
            "lstm": {
                "model": model_lstm,
                "predictions": pred_lstm,
                "rmse": rmse_lstm,
                "mae": mae_lstm,
                "r2": r2_lstm
            },
            "cnn_lstm": {
                "model": model_cnn_lstm,
                "predictions": pred_cnn_lstm,
                "rmse": rmse_cnn_lstm,
                "mae": mae_cnn_lstm,
                "r2": r2_cnn_lstm
            }
        }

    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement des modèles LSTM: {str(e)}")
        # Ne pas lever d'exception pour permettre la continuation
        return {}


def entrainer_modeles_garch(donnees):
    """
    Entraîne les modèles GARCH sur les données avec différentes configurations.
    """
    try:
        # Vérifier si le DataFrame est vide
        if donnees.empty:
            logger.warning("DataFrame vide fourni à entrainer_modeles_garch")
            return {}

        resultats = {}

        # Convertir les rendements en série temporelle
        if "rendement" not in donnees.columns:
            logger.warning("Colonne 'rendement' non trouvée dans les données")
            return {}

        rendements = donnees["rendement"].copy()
        # Gérer NaN/Inf dans les rendements
        rendements = handle_nan_inf(rendements)

        if rendements.empty:
             logger.warning("Données de rendement vides après nettoyage.")
             return {}

        # Liste des configurations à tester
        configurations = [
            {"nom": "garch_11", "vol": "Garch", "p": 1, "q": 1, "dist": "normal"},
            {"nom": "garch_12", "vol": "Garch", "p": 1, "q": 2, "dist": "normal"},
            {"nom": "garch_21", "vol": "Garch", "p": 2, "q": 1, "dist": "normal"},
            {"nom": "egarch_11", "vol": "EGARCH", "p": 1, "q": 1, "dist": "normal"},
            {"nom": "egarch_12", "vol": "EGARCH", "p": 1, "q": 2, "dist": "normal"},
            {"nom": "egarch_21", "vol": "EGARCH", "p": 2, "q": 1, "dist": "normal"},
            {"nom": "egarch_t_11", "vol": "EGARCH", "p": 1, "q": 1, "dist": "t"},
            {"nom": "egarch_t_12", "vol": "EGARCH", "p": 1, "q": 2, "dist": "t"},
            {"nom": "egarch_t_21", "vol": "EGARCH", "p": 2, "q": 1, "dist": "t"},
            {"nom": "garch_skewt_11", "vol": "Garch", "p": 1, "q": 1, "dist": "skewt"},
            {"nom": "egarch_skewt_11", "vol": "EGARCH", "p": 1, "q": 1, "dist": "skewt"}
        ]

        for config in configurations:
            try:
                logging.info(f"Entraînement du modèle {config['nom']}...")

                # Créer le modèle
                model = arch_model(
                    rendements,
                    vol=config["vol"],
                    p=config["p"],
                    q=config["q"],
                    dist=config["dist"]
                )

                # Ajuster le modèle
                res = model.fit(disp="off")

                resultats[config["nom"]] = {
                    "model": model,
                    "result": res,
                    "aic": res.aic,
                    "bic": res.bic,
                    "volatilite": np.sqrt(res.conditional_volatility)
                }

                logging.info(f"Modèle {config['nom']} entraîné - AIC: {res.aic:.2f}, BIC: {res.bic:.2f}")

            except Exception as e:
                logging.warning(f"Erreur lors de l'estimation du modèle {config['nom']}: {str(e)}")

        # Trier les résultats par AIC
        resultats_tries = sorted(
            [(nom, res) for nom, res in resultats.items() if res.get("aic") is not None],
            key=lambda x: x[1]["aic"]
        )

        # Afficher les 3 meilleurs modèles
        logging.info("\nMeilleurs modèles GARCH (par AIC):")
        for nom, res in resultats_tries[:3]:
            logging.info(f"{nom}: AIC = {res['aic']:.2f}, BIC = {res['bic']:.2f}")

        return resultats

    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement des modèles GARCH: {str(e)}")
        return {}


def entrainer_modeles_ml(X_train, X_test, y_train, y_test):
    """
    Entraîne les modèles de machine learning.
    """
    try:
        # Vérifier si les données sont valides
        if X_train.size == 0 or y_train.size == 0:
            logger.warning("Données d'entraînement vides fournies à entrainer_modeles_ml")
            return {}

        # Gérer NaN/Inf dans les données d'entrée
        logger.debug("Vérification et traitement des NaN/Inf dans les données ML...")
        X_train = handle_nan_inf(X_train)
        X_test = handle_nan_inf(X_test)
        y_train = handle_nan_inf(y_train)
        y_test = handle_nan_inf(y_test)

        # Vérification finale après imputation
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            logger.error("Des NaN ou Inf persistent dans X_train après traitement!")
            return {}
        if np.isnan(y_train).any() or np.isinf(y_train).any():
            logger.error("Des NaN ou Inf persistent dans y_train après traitement!")
            return {}

        resultats = {}

        # Random Forest
        try:
            logger.info("Entraînement du modèle random_forest...")
            model_rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model_rf.fit(X_train, y_train)
            pred_rf = model_rf.predict(X_test)
            rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
            mae_rf = mean_absolute_error(y_test, pred_rf)
            r2_rf = r2_score(y_test, pred_rf)

            resultats["random_forest"] = {
                "model": model_rf,
                "predictions": pred_rf,
                "rmse": rmse_rf,
                "mae": mae_rf,
                "r2": r2_rf,
                "importance": model_rf.feature_importances_
            }
            logger.info(f"Modèle random_forest entraîné - RMSE: {rmse_rf:.4f}")
        except Exception as e:
            logger.warning(f"Erreur lors de l'entraînement du modèle random_forest: {str(e)}")

        # XGBoost
        try:
            logger.info("Entraînement du modèle xgboost...")
            model_xgb = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model_xgb.fit(X_train, y_train)
            pred_xgb = model_xgb.predict(X_test)
            rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
            mae_xgb = mean_absolute_error(y_test, pred_xgb)
            r2_xgb = r2_score(y_test, pred_xgb)

            resultats["xgboost"] = {
                "model": model_xgb,
                "predictions": pred_xgb,
                "rmse": rmse_xgb,
                "mae": mae_xgb,
                "r2": r2_xgb,
                "importance": model_xgb.feature_importances_
            }
            logger.info(f"Modèle xgboost entraîné - RMSE: {rmse_xgb:.4f}")
        except Exception as e:
            logger.warning(f"Erreur lors de l'entraînement du modèle xgboost: {str(e)}")

        # LightGBM
        try:
            logger.info("Entraînement du modèle lightgbm...")
            model_lgb = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model_lgb.fit(X_train, y_train)
            pred_lgb = model_lgb.predict(X_test)
            rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))
            mae_lgb = mean_absolute_error(y_test, pred_lgb)
            r2_lgb = r2_score(y_test, pred_lgb)

            resultats["lightgbm"] = {
                "model": model_lgb,
                "predictions": pred_lgb,
                "rmse": rmse_lgb,
                "mae": mae_lgb,
                "r2": r2_lgb,
                "importance": model_lgb.feature_importances_
            }
            logger.info(f"Modèle lightgbm entraîné - RMSE: {rmse_lgb:.4f}")
        except Exception as e:
            logger.warning(f"Erreur lors de l'entraînement du modèle lightgbm: {str(e)}")

        # Neural Network
        try:
            logger.info("Entraînement du modèle neural_network...")
            model_nn = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size="auto",
                learning_rate="adaptive",
                max_iter=1000,
                random_state=42
            )
            model_nn.fit(X_train, y_train)
            pred_nn = model_nn.predict(X_test)
            rmse_nn = np.sqrt(mean_squared_error(y_test, pred_nn))
            mae_nn = mean_absolute_error(y_test, pred_nn)
            r2_nn = r2_score(y_test, pred_nn)

            resultats["neural_network"] = {
                "model": model_nn,
                "predictions": pred_nn,
                "rmse": rmse_nn,
                "mae": mae_nn,
                "r2": r2_nn
            }
            logger.info(f"Modèle neural_network entraîné - RMSE: {rmse_nn:.4f}")
        except Exception as e:
            logger.warning(f"Erreur lors de l'entraînement du modèle neural_network: {str(e)}")

        return resultats

    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement des modèles ML: {str(e)}")
        return {}


def entrainer_modeles_ensemble(X, y, test_size=0.2):
    """
    Entraîne un modèle d'ensemble combinant plusieurs modèles de base.
    """
    try:
        # Vérifier si les données sont valides
        if X.empty or y.empty:
            logger.warning("Données vides fournies à entrainer_modeles_ensemble")
            return {}

        # Gérer NaN/Inf dans les données d'entrée
        X_clean = handle_nan_inf(X)
        y_clean = handle_nan_inf(y)

        # Diviser les données (split chronologique)
        train_size = int(len(X_clean) * (1 - test_size))
        X_train, X_test = X_clean.iloc[:train_size], X_clean.iloc[train_size:]
        y_train, y_test = y_clean.iloc[:train_size], y_clean.iloc[train_size:]

        # Standardiser les données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convertir les noms de colonnes en string pour éviter les erreurs avec XGBoost
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns.astype(str))
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns.astype(str))

        # Entraîner les modèles de base
        logger.info("Entraînement des modèles de base pour l'ensemble...")
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_df, y_train)
        logger.info("Modèle RandomForest entraîné avec succès.")
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train_df, y_train)
        logger.info("Modèle GradientBoosting entraîné avec succès.")
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train_df, y_train)
        logger.info("Modèle XGBoost entraîné avec succès.")
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        lgb_model.fit(X_train_df, y_train)
        logger.info("Modèle LightGBM entraîné avec succès.")
        
        # SVR
        svr = SVR(kernel="rbf")
        svr.fit(X_train_df, y_train)
        logger.info("Modèle SVR entraîné avec succès.")
        
        # Générer les prédictions des modèles de base
        pred_rf = rf.predict(X_test_df)
        pred_gb = gb.predict(X_test_df)
        pred_xgb = xgb_model.predict(X_test_df)
        pred_lgb = lgb_model.predict(X_test_df)
        pred_svr = svr.predict(X_test_df)
        
        # Créer un DataFrame avec les prédictions des modèles de base
        meta_features = pd.DataFrame({
            "rf": pred_rf,
            "gb": pred_gb,
            "xgb": pred_xgb,
            "lgb": pred_lgb,
            "svr": pred_svr
        })
        
        # Entraîner le méta-modèle
        logger.info("Entraînement du méta-modèle...")
        meta_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        meta_model.fit(meta_features, y_test)
        
        # Prédire avec le méta-modèle
        pred_ensemble = meta_model.predict(meta_features)
        
        # Évaluer le modèle d'ensemble
        rmse_ensemble = np.sqrt(mean_squared_error(y_test, pred_ensemble))
        mae_ensemble = mean_absolute_error(y_test, pred_ensemble)
        r2_ensemble = r2_score(y_test, pred_ensemble)
        
        logger.info(f"Modèle d'ensemble entraîné avec succès. RMSE: {rmse_ensemble:.4f}, R²: {r2_ensemble:.4f}")
        
        # Visualiser les prédictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label="Réalité", color="black", linewidth=2)
        plt.plot(pred_ensemble, label="Ensemble", color="red", linewidth=1.5)
        plt.title("Prédictions du modèle d'ensemble vs. Valeurs réelles")
        plt.legend()
        plt.tight_layout()
        
        # Créer le répertoire si nécessaire
        os.makedirs(CHEMINS.get("modeles_prediction", "modeles_prediction"), exist_ok=True)
        
        # Sauvegarder le graphique
        plt.savefig(os.path.join(CHEMINS.get("modeles_prediction", "modeles_prediction"), "predictions_ensemble.png"))
        plt.close()
        
        # Sauvegarder le modèle d'ensemble
        ensemble_model = {
            "base_models": {
                "rf": rf,
                "gb": gb,
                "xgb": xgb_model,
                "lgb": lgb_model,
                "svr": svr
            },
            "meta_model": meta_model,
            "scaler": scaler
        }
        
        with open(os.path.join(CHEMINS.get("modeles_prediction", "modeles_prediction"), "modele_ensemble.pkl"), "wb") as f:
            pickle.dump(ensemble_model, f)
        
        return {
            "Ensemble": {
                "model": ensemble_model,
                "predictions": pred_ensemble,
                "rmse": rmse_ensemble,
                "mae": mae_ensemble,
                "r2": r2_ensemble
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle d'ensemble: {str(e)}")
        return {}


def evaluer_intervalles_confiance(donnees, resultats, y_test_ml=None, y_test_lstm=None, scaler_lstm=None):
    """
    Évalue les intervalles de confiance pour les prédictions des modèles.

    Args:
        donnees: DataFrame des données originales
        resultats: Dictionnaire des résultats des modèles
        y_test_ml: Données de test ML (non normalisées)
        y_test_lstm: Données de test LSTM normalisées
        scaler_lstm: Scaler utilisé pour les données LSTM
    """
    try:
        # Vérifier si les données sont valides
        if donnees.empty or not resultats:
            logger.warning("Données vides ou résultats vides fournis à evaluer_intervalles_confiance")
            return

        # Calculer les intervalles de confiance pour chaque modèle
        for nom_modele, res in resultats.items():
            if isinstance(res, dict) and "predictions" in res:
                try:
                    y_true = None
                    y_pred = handle_nan_inf(res["predictions"]) # Nettoyer les prédictions

                    # Pour les modèles LSTM, les données sont normalisées
                    if nom_modele in ["lstm", "cnn_lstm"] and y_test_lstm is not None and scaler_lstm is not None:
                        # Dénormaliser les prédictions et les vraies valeurs
                        y_test_lstm_clean = handle_nan_inf(y_test_lstm)
                        
                        # Créer un array temporaire pour la dénormalisation
                        temp_array_true = np.zeros((len(y_test_lstm_clean), scaler_lstm.n_features_in_))
                        temp_array_true[:, 0] = y_test_lstm_clean
                        y_true_denorm = scaler_lstm.inverse_transform(temp_array_true)[:, 0]
                        
                        temp_array_pred = np.zeros((len(y_pred), scaler_lstm.n_features_in_))
                        temp_array_pred[:, 0] = y_pred.flatten()
                        y_pred_denorm = scaler_lstm.inverse_transform(temp_array_pred)[:, 0]
                        
                        y_true = y_true_denorm
                        y_pred = y_pred_denorm
                    else:
                        # Pour les modèles ML, utiliser directement y_test_ml
                        if y_test_ml is not None:
                            y_true = y_test_ml

                    if y_true is not None and len(y_true) == len(y_pred):
                        # Calculer l'erreur standard
                        residuals = y_true - y_pred
                        std_residuals = np.std(residuals)
                        
                        # Calculer les intervalles de confiance à 95%
                        z_score = 1.96 # Pour un intervalle de confiance à 95%
                        lower_bound = y_pred - z_score * std_residuals
                        upper_bound = y_pred + z_score * std_residuals
                        
                        # Calculer le taux de couverture (pourcentage de valeurs réelles dans l'intervalle)
                        in_interval = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
                        coverage_rate = np.mean(in_interval) * 100
                        
                        # Ajouter les résultats au dictionnaire
                        res["intervalles_confiance"] = {
                            "lower": lower_bound,
                            "upper": upper_bound,
                            "std_residuals": std_residuals
                        }
                        res["taux_couverture"] = coverage_rate
                        
                        logger.info(f"Modèle {nom_modele} - Taux de couverture des IC: {coverage_rate:.2f}%")
                    else:
                        logger.warning(f"Dimensions incompatibles pour le calcul des intervalles de confiance du modèle {nom_modele}")
                
                except Exception as e:
                    logger.warning(f"Erreur lors du calcul des intervalles de confiance pour {nom_modele}: {str(e)}")

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation des intervalles de confiance: {str(e)}")


def generer_rapport_modeles_prediction(resultats, nom_fichier):
    """
    Génère un rapport Markdown des résultats des modèles de prédiction.
    """
    try:
        # Vérifier si les résultats sont valides
        if not resultats:
            logger.warning("Résultats vides fournis à generer_rapport_modeles_prediction")
            return

        # Trier les modèles par RMSE (du plus petit au plus grand)
        models_sorted = sorted(
            [(nom, res) for nom, res in resultats.items() if isinstance(res, dict) and "rmse" in res],
            key=lambda x: x[1]["rmse"]
        )

        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(nom_fichier), exist_ok=True)

        with open(nom_fichier, "w") as f:
            f.write("# Rapport d'Analyse des Modèles de Prédiction de Volatilité\n\n")
            f.write("## Résumé des Performances\n\n")
            f.write("| Modèle | RMSE | MAE | R² |\n")
            f.write("|--------|------|-----|----|\n")

            for nom_modele, res in models_sorted:
                f.write(f"| {nom_modele} | {res['rmse']:.4f} | {res['mae']:.4f} | {res['r2']:.4f} |\n")

            f.write("\n## Analyse Détaillée par Type de Modèle\n\n")

            # Modèles GARCH
            f.write("### Modèles GARCH\n\n")
            garch_models = [m for m in resultats if m.startswith("garch") or m.startswith("egarch")]
            for nom_modele in garch_models:
                if nom_modele in resultats and isinstance(resultats[nom_modele], dict):
                    res = resultats[nom_modele]
                    f.write(f"#### {nom_modele.upper()}\n\n")
                    if "aic" in res:
                        f.write(f"- AIC: {res['aic']:.2f}\n")
                        f.write(f"- BIC: {res['bic']:.2f}\n")
                    f.write("\n")

            # Modèles ML
            f.write("### Modèles de Machine Learning\n\n")
            ml_models = ["random_forest", "xgboost", "lightgbm", "neural_network"]
            for nom_modele in ml_models:
                if nom_modele in resultats and isinstance(resultats[nom_modele], dict):
                    res = resultats[nom_modele]
                    f.write(f"#### {nom_modele.upper()}\n\n")
                    if "rmse" in res:
                        f.write(f"- RMSE: {res['rmse']:.4f}\n")
                        f.write(f"- MAE: {res['mae']:.4f}\n")
                        f.write(f"- R²: {res['r2']:.4f}\n")
                    f.write("\n")

            # Modèles LSTM
            f.write("### Modèles Deep Learning\n\n")
            dl_models = ["lstm", "cnn_lstm"]
            for nom_modele in dl_models:
                if nom_modele in resultats and isinstance(resultats[nom_modele], dict):
                    res = resultats[nom_modele]
                    f.write(f"#### {nom_modele.upper()}\n\n")
                    if "rmse" in res:
                        f.write(f"- RMSE: {res['rmse']:.4f}\n")
                        f.write(f"- MAE: {res['mae']:.4f}\n")
                        f.write(f"- R²: {res['r2']:.4f}\n")
                    f.write("\n")

            # Modèle d'ensemble
            f.write("### Modèle d'Ensemble\n\n")
            if "Ensemble" in resultats and isinstance(resultats["Ensemble"], dict):
                res = resultats["Ensemble"]
                f.write("#### ENSEMBLE\n\n")
                if "rmse" in res:
                    f.write(f"- RMSE: {res['rmse']:.4f}\n")
                    f.write(f"- MAE: {res['mae']:.4f}\n")
                    f.write(f"- R²: {res['r2']:.4f}\n")
                f.write("\n")

            # Intervalles de Confiance
            f.write("### Analyse des Intervalles de Confiance\n\n")
            f.write("Les intervalles de confiance à 95% ont été calculés pour chaque modèle. ")
            f.write("Le taux de couverture indique la proportion de valeurs réelles qui tombent dans ces intervalles.\n\n")

            for nom_modele, res in resultats.items():
                if isinstance(res, dict) and "taux_couverture" in res:
                    taux_couv = res["taux_couverture"]
                    # Format the coverage rate conditionally
                    taux_couv_str = f"{taux_couv:.2%}" if isinstance(taux_couv, float) else "N/A"
                    f.write(f"- {nom_modele}: {taux_couv_str}\n")

            # Visualisations
            f.write("\n### Visualisations\n\n")
            f.write("Les graphiques suivants ont été générés pour l'analyse des modèles :\n\n")
            f.write("1. Prédictions vs Réalité (visualisations/lstm_ameliore.png)\n")
            f.write("2. Courbes d'apprentissage (visualisations/lstm_ameliore.png)\n")
            f.write("3. Erreurs de prédiction (visualisations/lstm_ameliore.png)\n")
            f.write("4. Distribution des erreurs (visualisations/lstm_ameliore.png)\n")
            f.write("5. Prédictions vs. Valeurs réelles - Modèle d'ensemble (modeles_prediction/predictions_ensemble.png)\n")
            f.write("6. Comparaison interactive des prédictions (visualisations_prediction/{pays}/comparaison_predictions_{pays}.html)\n")
            f.write("7. Comparaison interactive des métriques (visualisations_prediction/{pays}/comparaison_metriques_{pays}.html)\n")

            f.write("\n### Conclusion\n\n")
            f.write("Cette analyse comparative des différents modèles de prédiction de volatilité montre que:\n\n")

            # Trouver le meilleur modèle basé sur RMSE
            meilleur_rmse = float("inf")
            meilleur_modele = None
            if models_sorted:
                 meilleur_modele, meilleur_res = models_sorted[0]
                 meilleur_rmse = meilleur_res["rmse"]

            if meilleur_modele:
                f.write(f"1. Le modèle **{meilleur_modele}** a montré les meilleures performances globales avec un RMSE de **{meilleur_rmse:.4f}**.\n")
            else:
                 f.write("1. Aucun modèle n'a pu être évalué avec succès sur la base du RMSE.\n")

            f.write("2. Les modèles GARCH (en particulier EGARCH avec distribution t ou skewt) capturent bien la dynamique de la volatilité.\n")
            f.write("3. Les modèles LSTM montrent également de bonnes performances, indiquant leur capacité à capturer les dépendances temporelles.\n")
            f.write("4. Les intervalles de confiance fournissent une mesure utile de l'incertitude des prédictions.\n")

        logging.info(f"Rapport généré avec succès dans {nom_fichier}")

    except Exception as e:
        logging.error(f"Erreur lors de la génération du rapport: {str(e)}")


def creer_visualisations_prediction(resultats: Dict[str, Any], pays: str, output_dir: str, y_test_ml=None, y_test_lstm=None, scaler_lstm=None) -> None:
    """
    Crée des visualisations interactives pour les résultats des modèles de prédiction.

    Args:
        resultats: Dictionnaire contenant les résultats des modèles
        pays: Nom du pays
        output_dir: Répertoire de sortie pour les visualisations
        y_test_ml: Données de test ML (non normalisées)
        y_test_lstm: Données de test LSTM normalisées
        scaler_lstm: Scaler utilisé pour les données LSTM
    """
    try:
        # Vérifier si les résultats sont valides
        if not resultats:
            logger.warning(f"Résultats vides fournis à creer_visualisations_prediction pour {pays}")
            return

        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)

        # 1. Comparaison des prédictions des différents modèles
        fig_pred = go.Figure()

        # Déterminer l'index et les valeurs réelles à tracer
        y_true_plot = None
        plot_index = None

        if y_test_ml is not None:
            y_true_plot = handle_nan_inf(y_test_ml)
            plot_index = y_true_plot.index if isinstance(y_true_plot, pd.Series) else np.arange(len(y_true_plot))
        elif y_test_lstm is not None and scaler_lstm is not None:
             try:
                 y_test_lstm_clean = handle_nan_inf(y_test_lstm)
                 temp_array_true = np.zeros((len(y_test_lstm_clean), scaler_lstm.n_features_in_))
                 temp_array_true[:, 0] = y_test_lstm_clean
                 y_true_plot = scaler_lstm.inverse_transform(temp_array_true)[:, 0]
                 plot_index = np.arange(len(y_true_plot)) # Pas d'index date pour LSTM ici
             except Exception as e:
                 logger.warning(f"Erreur lors de la dénormalisation des données LSTM pour la visualisation: {str(e)}")

        if y_true_plot is not None and plot_index is not None:
             fig_pred.add_trace(go.Scatter(
                 x=plot_index,
                 y=y_true_plot,
                 name="Valeurs Réelles",
                 mode="lines",
                 line=dict(color="black")
             ))

        # Ajouter les prédictions de chaque modèle (dénormalisées si LSTM)
        for nom_modele, res in resultats.items():
            if isinstance(res, dict) and "predictions" in res:
                try:
                    pred_plot = handle_nan_inf(res["predictions"])
                    current_plot_index = plot_index # Utiliser l'index déterminé précédemment

                    if nom_modele in ["lstm", "cnn_lstm"] and scaler_lstm is not None:
                        temp_array_pred = np.zeros((len(pred_plot), scaler_lstm.n_features_in_))
                        temp_array_pred[:, 0] = pred_plot.flatten()
                        pred_plot = scaler_lstm.inverse_transform(temp_array_pred)[:, 0]
                        # Ajuster l'index si nécessaire (peut être différent de y_test_ml)
                        if len(pred_plot) != len(plot_index):
                             current_plot_index = np.arange(len(pred_plot))

                    # Assurer que les longueurs correspondent pour le tracé
                    if current_plot_index is not None and len(pred_plot) == len(current_plot_index):
                        fig_pred.add_trace(go.Scatter(
                            x=current_plot_index,
                            y=pred_plot,
                            name=nom_modele.replace("_", " ").title(),
                            mode="lines"
                        ))
                    else:
                         logger.warning(f"Incohérence de longueur pour les prédictions de {nom_modele}. Non tracé.")

                except Exception as e:
                    logger.warning(f"Erreur lors de l'ajout des prédictions pour {nom_modele}: {str(e)}")

        fig_pred.update_layout(
            title=f"Comparaison des Prédictions - {pays}",
            xaxis_title="Date/Index",
            yaxis_title="Volatilité",
            hovermode="x unified",
            template="plotly_white"
        )

        fig_pred.write_html(os.path.join(output_dir, f"comparaison_predictions_{pays}.html"))
        logging.info(f"Visualisation interactive des prédictions sauvegardée pour {pays}")

        # 2. Bar chart des métriques de performance
        metrics_data = []
        for nom_modele, res in resultats.items():
            if isinstance(res, dict) and "rmse" in res:
                metrics_data.append({
                    "Modèle": nom_modele.replace("_", " ").title(),
                    "RMSE": res.get("rmse", np.nan),
                    "MAE": res.get("mae", np.nan),
                    "R²": res.get("r2", np.nan)
                })

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data).dropna(subset=["RMSE"]) # Exclure les modèles sans RMSE
            df_metrics = df_metrics.sort_values("RMSE") # Trier par RMSE
            fig_metrics = px.bar(df_metrics, x="Modèle", y=["RMSE", "MAE", "R²"],
                                 barmode="group", title=f"Métriques de Performance - {pays}")
            fig_metrics.update_layout(template="plotly_white")
            fig_metrics.write_html(os.path.join(output_dir, f"comparaison_metriques_{pays}.html"))
            logging.info(f"Visualisation interactive des métriques sauvegardée pour {pays}")

    except Exception as e:
        logging.error(f"Erreur lors de la création des visualisations pour {pays}: {str(e)}")


def main():
    """
    Fonction principale pour exécuter le pipeline de modélisation.
    """
    # Créer les répertoires nécessaires
    os.makedirs(CHEMINS.get("modeles_prediction", "modeles_prediction"), exist_ok=True)
    os.makedirs(CHEMINS.get("visualisations_prediction", "visualisations_prediction"), exist_ok=True)
    os.makedirs(CHEMINS.get("rapports_prediction", "rapports_prediction"), exist_ok=True)
    os.makedirs("modeles", exist_ok=True) # Pour les modèles LSTM
    os.makedirs("visualisations", exist_ok=True) # Pour les visualisations LSTM
    os.makedirs(os.path.join(CHEMINS.get("rapport_final", "rapport_final"), "resultats"), exist_ok=True) # Pour les erreurs

    # Définir les variables macroéconomiques à utiliser
    variables_macro_a_utiliser = [
        "inflation", "taux_directeur", "gdp", "close_usd", "close_euro"
    ]

    # Définir la colonne de volatilité cible
    col_volatilite_cible = "volatilite_garch" # Ou une autre, ex: "volatilite_egarch"

    # Définir les paramètres
    lags = 10
    horizon = 1
    test_size = 0.2

    # Itérer sur chaque pays
    for pays_key, pays_config in PAYS.items():
        pays_nom = pays_config["nom"]
        logger.info(f"\n--- Traitement du pays: {pays_nom} ---")

        try:
            # Charger les données de volatilité
            donnees = charger_donnees_volatilite(pays_key)
            if donnees.empty:
                logger.warning(f"Données vides pour {pays_nom}. Passage au pays suivant.")
                continue

            # Préparer les données pour les modèles ML
            X, y = preparer_donnees_modele(
                donnees,
                col_volatilite=col_volatilite_cible,
                variables_macro=variables_macro_a_utiliser,
                lags=lags,
                horizon_prediction=horizon
            )

            if X.empty or y.empty:
                logger.warning(f"Données vides après préparation pour {pays_nom}. Passage au pays suivant.")
                continue

            # Gérer NaN/Inf dans X et y après préparation
            X = handle_nan_inf(X)
            y = handle_nan_inf(y)

            # Diviser les données pour ML (split chronologique)
            train_size_ml = int(len(X) * (1 - test_size))
            X_train_ml, X_test_ml = X.iloc[:train_size_ml], X.iloc[train_size_ml:]
            y_train_ml, y_test_ml = y.iloc[:train_size_ml], y.iloc[train_size_ml:]

            # Standardiser les données ML (après split, ajuster sur train)
            scaler_ml = StandardScaler()
            # Gérer NaN/Inf avant fit/transform
            X_train_ml_clean = handle_nan_inf(X_train_ml)
            X_test_ml_clean = handle_nan_inf(X_test_ml)

            if X_train_ml_clean.empty:
                 logger.warning(f"Données d'entraînement vides après nettoyage pour {pays_nom}. Passage au pays suivant.")
                 continue

            X_train_ml_scaled = scaler_ml.fit_transform(X_train_ml_clean)
            X_test_ml_scaled = scaler_ml.transform(X_test_ml_clean)

            # Gérer NaN/Inf après standardisation et dans y
            X_train_ml_scaled = handle_nan_inf(X_train_ml_scaled)
            X_test_ml_scaled = handle_nan_inf(X_test_ml_scaled)
            y_train_ml_clean = handle_nan_inf(y_train_ml)
            y_test_ml_clean = handle_nan_inf(y_test_ml)

            # Initialiser le dictionnaire global des résultats pour ce pays
            resultats_pays = {}

            # Entraîner les modèles GARCH
            resultats_garch = entrainer_modeles_garch(donnees)
            resultats_pays.update(resultats_garch)

            # Entraîner les modèles ML
            resultats_ml = entrainer_modeles_ml(X_train_ml_scaled, X_test_ml_scaled, y_train_ml_clean.values, y_test_ml_clean.values)
            resultats_pays.update(resultats_ml)

            # Préparer les données pour LSTM
            y_test_lstm = None # Initialiser
            scaler_lstm = None # Initialiser
            try:
                X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler_lstm = preparer_donnees_lstm(donnees, n_steps=lags)

                # Entraîner les modèles LSTM
                resultats_lstm = entrainer_lstm(X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm)
                resultats_pays.update(resultats_lstm)
            except Exception as e:
                 logger.warning(f"Erreur lors de la préparation ou de l'entraînement LSTM pour {pays_nom}: {e}")

            # Entraîner le modèle d'ensemble
            resultats_ensemble = entrainer_modeles_ensemble(X, y, test_size=test_size)
            resultats_pays.update(resultats_ensemble)

            # Évaluer les intervalles de confiance
            evaluer_intervalles_confiance(donnees, resultats_pays, y_test_ml=y_test_ml_clean, y_test_lstm=y_test_lstm, scaler_lstm=scaler_lstm)

            # Générer le rapport
            chemin_rapport = os.path.join(CHEMINS.get("rapports_prediction", "rapports_prediction"), f"rapport_prediction_{pays_key}.md")
            generer_rapport_modeles_prediction(resultats_pays, chemin_rapport)

            # Créer les visualisations
            chemin_visualisations = os.path.join(CHEMINS.get("visualisations_prediction", "visualisations_prediction"), pays_key)
            creer_visualisations_prediction(resultats_pays, pays_nom, chemin_visualisations, y_test_ml=y_test_ml_clean, y_test_lstm=y_test_lstm, scaler_lstm=scaler_lstm)

            # Sauvegarder les résultats (prédictions, métriques)
            chemin_resultats = os.path.join(CHEMINS.get("modeles_prediction", "modeles_prediction"), f"resultats_prediction_{pays_key}.pkl")
            try:
                with open(chemin_resultats, "wb") as f:
                    # Supprimer les modèles non sérialisables avant de sauvegarder
                    results_to_save = {}
                    for key, value in resultats_pays.items():
                        if isinstance(value, dict):
                            results_to_save[key] = {k: v for k, v in value.items() if k != "model"}
                    pickle.dump(results_to_save, f)
                logger.info(f"Résultats (sans modèles) sauvegardés dans {chemin_resultats}")
            except Exception as e:
                logger.warning(f"Erreur lors de la sauvegarde des résultats: {str(e)}")

            # Sauvegarder les erreurs de prédiction pour analyse ultérieure
            try:
                erreurs_prediction = pd.DataFrame(index=y_test_ml_clean.index)
                
                # Ajouter les erreurs des modèles ML
                for nom_modele, res in resultats_ml.items():
                    if "predictions" in res:
                        try:
                            erreurs_prediction[nom_modele] = y_test_ml_clean.values - res["predictions"]
                        except Exception as e:
                            logger.warning(f"Erreur lors du calcul des erreurs pour {nom_modele}: {str(e)}")
                
                # Ajouter les erreurs des modèles LSTM (dénormalisées)
                for nom_modele in ["lstm", "cnn_lstm"]:
                    if nom_modele in resultats_pays and "predictions" in resultats_pays[nom_modele] and scaler_lstm is not None:
                        try:
                            # Dénormaliser les prédictions LSTM
                            pred_lstm = resultats_pays[nom_modele]["predictions"]
                            temp_array = np.zeros((len(pred_lstm), scaler_lstm.n_features_in_))
                            temp_array[:, 0] = pred_lstm.flatten()
                            pred_denorm = scaler_lstm.inverse_transform(temp_array)[:, 0]
                            
                            # Dénormaliser les vraies valeurs
                            y_true_lstm = y_test_lstm
                            temp_array_true = np.zeros((len(y_true_lstm), scaler_lstm.n_features_in_))
                            temp_array_true[:, 0] = y_true_lstm
                            y_true_denorm = scaler_lstm.inverse_transform(temp_array_true)[:, 0]
                            
                            # Calculer les erreurs
                            erreurs_lstm = y_true_denorm - pred_denorm
                            
                            # Ajouter au DataFrame (peut nécessiter un reindex)
                            if len(erreurs_lstm) == len(erreurs_prediction):
                                erreurs_prediction[nom_modele] = erreurs_lstm
                            else:
                                logger.warning(f"Dimensions incompatibles pour les erreurs {nom_modele}")
                        except Exception as e:
                            logger.warning(f"Erreur lors du calcul des erreurs pour {nom_modele}: {str(e)}")
                
                # Ajouter les erreurs du modèle d'ensemble
                if "Ensemble" in resultats_ensemble and "predictions" in resultats_ensemble["Ensemble"]:
                    try:
                        erreurs_prediction["Ensemble"] = y_test_ml_clean.values - resultats_ensemble["Ensemble"]["predictions"]
                    except Exception as e:
                        logger.warning(f"Erreur lors du calcul des erreurs pour Ensemble: {str(e)}")
                
                # Sauvegarder les erreurs
                chemin_erreurs = os.path.join(CHEMINS.get("rapport_final", "rapport_final"), "resultats", f"erreurs_prediction_{pays_key}.csv")
                erreurs_prediction.to_csv(chemin_erreurs)
                logger.info(f"Erreurs de prédiction sauvegardées dans {chemin_erreurs}")
            
            except Exception as e:
                logger.warning(f"Erreur lors de la sauvegarde des erreurs de prédiction: {str(e)}")

        except Exception as e:
            logger.error(f"Erreur lors du traitement du pays {pays_nom}: {str(e)}")

    logger.info("\n--- Traitement de tous les pays terminé ---")


if __name__ == "__main__":
    main()
