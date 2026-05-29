# Architecture Decision Record — Projet CPR
## Prédiction de churn et recommandation d'offre

**Namespace :** `projet-CPR`  
**Quota :** 2500m CPU / 1.5Gi mémoire  
**Date :** Mai 2026

---

## Choix du cas et compatibilité avec le quota

Le cas retenu est le Cas 3 : Prédiction de churn et recommandation d'offre. Ce choix repose sur la compatibilité directe entre la nature des modèles impliqués et le quota imposé de 2500m CPU et 1.5Gi de mémoire. Les deux modèles à déployer sont un classifieur XGBoost pour le score de churn et un classifieur multi-classes pour la recommandation d'offre. Ces deux artefacts sont très légers, de l'ordre de 20Mo chacun. En mémoire, le service d'inférence consomme principalement au démarrage lors du chargement des librairies Python, xgboost et scikit-learn, soit environ 240Mi au total — les modèles eux-mêmes ne représentent qu'une fraction négligeable de cette consommation.

L'estimation mémoire par service est la suivante. Le service de preprocessing, qui ne charge aucun modèle et effectue uniquement des transformations sur des payloads JSON légers, est estimé à 128Mi de request et 256Mi de limit. Le service d'inférence, qui charge les deux modèles au démarrage et les conserve en mémoire pour toute la durée de vie du pod, est estimé à 300Mi de request et 512Mi de limit. Le service de monitoring, qui ne fait que comptabiliser des métriques en mémoire, est estimé à 64Mi de request et 128Mi de limit. La somme des requests mémoire s'établit donc à 492Mi, ce qui représente 32% du quota de 1536Mi — une marge très confortable qui absorbe les variations de charge sans risque d'OOMKill.

Sur le plan CPU, le défi spécifique de ce cas est la tenue à 150 req/min au niveau stress. Un service Flask mono-threadé saturerait à ce niveau. Le service d'inférence sera donc servi par Gunicorn, avec un nombre de workers dimensionné selon la règle `(2 × nb_cores) + 1`. Avec 700m CPU alloués en request (soit 0.7 core), cela donne 2 workers Gunicorn, suffisant pour absorber 150 req/min si chaque inférence XGBoost reste sous 100ms, ce qui est attendu pour ce type de modèle. La somme des requests CPU s'établit à 850m sur 2500m alloués.

---

## Dataset

Le dataset principal retenu est le Telco Customer Churn dataset publié par IBM, composé de 7 043 clients décrits par 21 variables couvrant les caractéristiques du contrat, les services souscrits et le comportement de paiement. Ce dataset est disponible publiquement sur Kaggle, l'UCI ML Repository et GitHub, sous licence publique, sans restriction d'usage. Son format CSV d'environ 1Mo le rend trivial à versionner directement dans le repo sous `data/churn.csv`.

Le modèle de recommandation d'offre ne peut pas être entraîné sur ce dataset tel quel, car il ne contient pas de variable cible correspondant à une offre commerciale. Des données synthétiques seront donc générées à partir des mêmes variables features, en assignant à chaque profil client une offre parmi cinq catégories fictives : remise_tarifaire, upgrade_forfait, offre_fidelite, option_gratuite et maintien_standard. Cette génération est scriptée et reproductible, versionnée dans le repo sous `data/generate_synthetic.py`, de sorte que le dataset synthétique peut être recréé à tout moment depuis les données originales.

---

## Communication inter-services et architecture des modèles

L'architecture retenue comprend trois services distincts déployés dans le namespace `projet-CPR` : un service de preprocessing, un service d'inférence principal et un service de monitoring. Chaque service expose un objet Kubernetes de type ClusterIP, ce qui lui attribue un nom DNS stable et interne au cluster, indépendant de l'IP éphémère de son pod. Les appels entre services transitent exclusivement par ce réseau interne et n'exposent aucun port vers l'extérieur, à l'exception du service d'inférence qui est rendu accessible depuis l'extérieur du cluster via `minikube service` pour recevoir les requêtes du script de charge.

Le flux d'une requête est le suivant. Le script de charge envoie un profil client en JSON au service d'inférence via l'URL publique exposée par Minikube. Le service d'inférence délègue le preprocessing au service dédié via `http://preprocessing-svc:8001/process`, récupère les features transformées, exécute séquentiellement le modèle de churn puis, si le score dépasse le seuil configurable, le modèle de recommandation. La réponse JSON est ensuite transmise en parallèle au service de monitoring via `http://monitoring-svc:8002/log` avant d'être retournée au client.

Les deux modèles sont hébergés dans le même service d'inférence plutôt que dans deux services séparés. Ce choix est justifié par trois arguments. Premièrement, les deux artefacts sont légers (~20Mo chacun) et leur chargement combiné ne représente pas une contrainte mémoire significative. Deuxièmement, les deux modèles sont appelés séquentiellement dans une logique de cascade — le second n'est invoqué que si le premier dépasse un seuil — ce qui rend la séparation en deux services inutilement coûteuse en latence réseau. Troisièmement, dans le cadre du quota de 1.5Gi, déployer un quatrième service dédié au second modèle consommerait du quota supplémentaire sans apporter de bénéfice architectural réel à ce stade. Cette décision serait à réévaluer si les cycles de mise à jour des deux modèles devenaient indépendants en production.

---

## Outil CI/CD

L'outil CI/CD retenu est GitHub Actions. Ce choix repose sur deux critères concrets. Premièrement, GitHub Actions est nativement intégré au dépôt Git sans nécessiter de service externe à configurer ou maintenir : les pipelines se déclenchent directement sur les événements du dépôt (push sur main, pull request), et les secrets nécessaires au push des images Docker Hub sont gérés directement dans les paramètres du dépôt, sans infrastructure supplémentaire. Deuxièmement, l'écosystème d'actions préconstruites couvre exactement les besoins du projet : `docker/build-push-action` pour construire et pousser les images sur Docker Hub, et `actions/setup-python` pour exécuter les tests avec couverture. Cela réduit le pipeline à une configuration déclarative en YAML sans écrire de scripts shell complexes.

Le pipeline est structuré en deux jobs séquentiels. Le premier job exécute les tests unitaires avec pytest et vérifie que la couverture atteint le seuil de 80% — si ce seuil n'est pas atteint, le pipeline échoue et aucune image n'est construite ni poussée. Le second job, conditionnel au succès du premier, construit les trois images Docker avec les dépendances pinned et les pousse sur Docker Hub avec un tag de version correspondant au SHA du commit. Ce comportement garantit qu'aucune image non testée ne peut atteindre le registry.

---

## Stratégie de déploiement

La stratégie de déploiement retenue est RollingUpdate avec `maxSurge: 1` et `maxUnavailable: 0`. Ce choix garantit l'absence de downtime lors des mises à jour : Kubernetes démarre le nouveau pod avant de terminer l'ancien, maintenant le service disponible en permanence. Cependant, cette stratégie exige que le quota dispose d'une marge suffisante pour héberger temporairement un pod supplémentaire du service mis à jour.

Le calcul de cette marge est le suivant. La somme des requests déclarées par les trois services en fonctionnement normal s'établit à 850m CPU et 492Mi mémoire. Le quota alloué au namespace est de 2500m CPU et 1536Mi mémoire. La marge disponible est donc de 1650m CPU et 1044Mi mémoire. Le service le plus lourd est le service d'inférence, avec 700m CPU et 300Mi mémoire en request. Avec `maxSurge: 1`, Kubernetes démarre un pod d'inférence supplémentaire pendant la transition, ce qui requiert 700m CPU et 300Mi mémoire supplémentaires. La marge disponible de 1650m CPU et 1044Mi mémoire est largement supérieure à ces besoins dans les deux dimensions. Le RollingUpdate est donc viable sans risque de blocage en état Pending.

Le paramètre `maxUnavailable: 0` est retenu car la contrainte métier du cas impose une latence inférieure à 200ms et une disponibilité continue du service de prédiction. Accepter un downtime même bref lors d'une mise à jour serait incompatible avec cet objectif. Si le quota venait à être réduit au point que la marge deviendrait insuffisante pour le surge, la stratégie serait basculée en Recreate, avec le downtime que cela implique mais sans ressource supplémentaire nécessaire.

---

## Tableau récapitulatif des ressources

| Service          | CPU request | Memory request | CPU limit | Memory limit |
|------------------|-------------|----------------|-----------|--------------|
| preprocessing    | 100m        | 128Mi          | 200m      | 256Mi        |
| inference        | 700m        | 300Mi          | 1200m     | 512Mi        |
| monitoring       | 50m         | 64Mi           | 100m      | 128Mi        |
| **TOTAL**        | **850m**    | **492Mi**      | **1500m** | **896Mi**    |
| **QUOTA**        | **2500m**   | **1536Mi**     | **2500m** | **1536Mi**   |
| **MARGE**        | **1650m**   | **1044Mi**     | **1000m** | **640Mi**    |


---

## Multi-stage build — service inference

L'image du service d'inférence dépasse 1Go décompressée (1.54Go mesuré
avec docker images). Un multi-stage build a été implémenté conformément
aux exigences du TP. La réduction de taille est cependant minimale
(1.54Go → 1.53Go) car les dépendances ML (scikit-learn, xgboost, numpy,
pandas) sont distribuées sous forme de wheels pré-compilées sur PyPI —
aucune compilation avec gcc/g++ n'a lieu, donc aucun outil de build
n'est présent dans l'image finale à supprimer. La taille incompressible
est celle des librairies ML elles-mêmes, toutes nécessaires au runtime.
