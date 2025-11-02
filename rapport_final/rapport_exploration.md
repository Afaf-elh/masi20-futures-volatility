# Rapport d'exploration des données

Date: 2025-11-02 20:11:09

## Résumé global

- Nombre de pays analysés: 4
- Nombre total de fichiers: 7

## Détails par pays

### Maroc

- Nombre de fichiers: 1

| Fichier | Lignes | Colonnes | Plage de dates | Valeurs manquantes | Valeurs aberrantes | Stationnarité |
|---------|--------|----------|----------------|-------------------|-------------------|---------------|
| Masi20.csv | 1287 | 6 | 1970-01-01 à 2025-02-25 | 1287 (16.67%) | 24 (1.86%) | 0/5 colonnes |

### Vietnam

- Nombre de fichiers: 2

| Fichier | Lignes | Colonnes | Plage de dates | Valeurs manquantes | Valeurs aberrantes | Stationnarité |
|---------|--------|----------|----------------|-------------------|-------------------|---------------|
| VN30.csv | 2568 | 6 | 1970-01-01 à 2025-02-25 | 2568 (16.67%) | Aucune | 0/5 colonnes |
| VN30_Future.csv | 1893 | 6 | 1970-01-01 à 2025-03-11 | Aucune | 22 (1.16%) | 1/5 colonnes |

### Afrique

- Nombre de fichiers: 2

| Fichier | Lignes | Colonnes | Plage de dates | Valeurs manquantes | Valeurs aberrantes | Stationnarité |
|---------|--------|----------|----------------|-------------------|-------------------|---------------|
| JSE40.csv | 2524 | 7 | 2015-01-02 à 2025-02-05 | 36 (0.20%) | Aucune | N/A |
| JSE40_Future.csv | 2551 | 7 | 2015-01-02 à 2025-02-28 | 62 (0.35%) | Aucune | N/A |

### India

- Nombre de fichiers: 2

| Fichier | Lignes | Colonnes | Plage de dates | Valeurs manquantes | Valeurs aberrantes | Stationnarité |
|---------|--------|----------|----------------|-------------------|-------------------|---------------|
| Nifty50.csv | 2518 | 6 | 1970-01-01 à 2025-02-25 | Aucune | 34 (1.35%) | 0/5 colonnes |
| Nifty50_Future.csv | 2624 | 7 | 2015-01-01 à 2025-02-28 | 106 (0.58%) | Aucune | N/A |

## Problèmes identifiés

### Maroc

- **Masi20.csv**: 100.00% de valeurs manquantes dans la colonne 'Volume'

### Vietnam

- **VN30.csv**: 100.00% de valeurs manquantes dans la colonne 'Volume'

## Recommandations

1. **Traitement des valeurs manquantes**: Utiliser des méthodes d'interpolation pour combler les valeurs manquantes dans les séries temporelles.
2. **Harmonisation des formats de date**: Standardiser tous les formats de date au format AAAA-MM-JJ.
3. **Traitement des valeurs aberrantes**: Examiner et éventuellement corriger les valeurs aberrantes identifiées.
4. **Alignement des périodes**: S'assurer que toutes les séries temporelles couvrent la même période pour faciliter l'analyse comparative.
5. **Conversion à la fréquence journalière**: Convertir toutes les données à une fréquence journalière pour l'analyse de la volatilité.
