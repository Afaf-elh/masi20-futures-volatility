"""
Script principal pour l'exécution de l'analyse complète de la volatilité des futures sur MASI20
et benchmarking avec d'autres marchés émergents.
Période d'étude: 2015-01-01 à 2024-12-31
"""

import os
import sys
import time
import subprocess

def executer_script(script_path, description):
    """
    Exécute un script Python et affiche sa progression.
    
    Args:
        script_path: Chemin vers le script à exécuter
        description: Description de l'étape
    """
    print(f"\n{'='*80}")
    print(f"ÉTAPE: {description}")
    print(f"{'='*80}\n")
    
    try:
        # Exécuter le script
        process = subprocess.Popen(['python3', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # Afficher la sortie en temps réel
        for line in process.stdout:
            print(line, end='')
        
        # Attendre la fin de l'exécution
        process.wait()
        
        # Vérifier si l'exécution a réussi
        if process.returncode == 0:
            print(f"\n✅ {description} terminé avec succès.")
        else:
            print(f"\n❌ {description} a échoué avec le code de retour {process.returncode}.")
            # Afficher les erreurs
            for line in process.stderr:
                print(line, end='')
    
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution de {script_path}: {str(e)}")

def main():
    """
    Fonction principale pour exécuter l'analyse complète.
    """
    print("\n" + "*"*100)
    print("*" + " "*98 + "*")
    print("*" + " "*20 + "ANALYSE DE LA VOLATILITÉ DES FUTURES SUR MASI20 ET BENCHMARKING" + " "*20 + "*")
    print("*" + " "*98 + "*")
    print("*" + " "*30 + "Période d'étude: 2015-01-01 à 2024-12-31" + " "*30 + "*")
    print("*" + " "*98 + "*")
    print("*"*100 + "\n")
    
    # Créer les répertoires nécessaires
    os.makedirs('data_harmonisee', exist_ok=True)
    os.makedirs('volatilite', exist_ok=True)
    os.makedirs('analyse_impact', exist_ok=True)
    os.makedirs('analyse_comparative', exist_ok=True)
    os.makedirs('modeles_prediction', exist_ok=True)
    os.makedirs('rapport_final', exist_ok=True)
    os.makedirs('rapport_final/visualisations', exist_ok=True)
    
    # Définir les étapes de l'analyse
    etapes = [
        ('1_exploration_donnees_ameliore.py', "Exploration et compréhension des données"),
        ('2_harmonisation_donnees_ameliore.py', "Harmonisation et préparation des données"),
        ('3_calcul_volatilite_simulation_futures_ameliore.py', "Calcul de la volatilité des indices et simulation des futures sur MASI20"),
        ('4_analyse_impact_variables_macro_ameliore.py', "Analyse de l'impact des variables macroéconomiques"),
        ('5_analyse_comparative_ameliore.py', "Analyse comparative avec d'autres marchés émergents"),
        ('6_modeles_prediction_avances_ameliore.py', "Développement et évaluation des modèles de prédiction"),
        ('7_visualisations_erreur_ameliore.py', "Génération des visualisations des erreurs")
    ]
    
    # Exécuter chaque étape
    for script_path, description in etapes:
        executer_script(script_path, description)
        time.sleep(1)  # Pause entre les étapes
    
    print("\n" + "*"*100)
    print("*" + " "*98 + "*")
    print("*" + " "*30 + "ANALYSE COMPLÈTE TERMINÉE AVEC SUCCÈS" + " "*30 + "*")
    print("*" + " "*98 + "*")
    print("*" + " "*15 + "Les résultats sont disponibles dans le dossier 'rapport_final'" + " "*15 + "*")
    print("*" + " "*98 + "*")
    print("*"*100 + "\n")

if __name__ == "__main__":
    main()
