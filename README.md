# Projet IA et biodiversité 


# Objectif de notre projet

Notre projet consiste en la réalisation d'un modèle CNN pour la classification d'espèces de plantes. Nous nous basons sur le dataset plantnet300k afin de créer notre modèle. Dans un deuxième temps nous allons comparer les différents modèles existant et créer une interface web qui prendra une image en entrée et qui donnera les 5 prédictions les plus probables à l'aide d'un modèle préalablement sélectionné.

# Fichiers nécessaires

Pour pouvoir éxécuter ces différents codes vous allez avoir besoin de 'quelques' fichiers.

- Le dataset plantnet300k (https://zenodo.org/record/5645731#.Yuehg3ZBxPY)
- Certains modèles préentrainés (https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/)


# Installation générale

- Cloner le fichier à partir de git (https://github.com/ClementTh/IA_biodiveristy)
- Télécharger les différents fichiers demandés (Voir Fichiers nécessaires)
- Exécuter le requierments.txt

# Mise en place pour le site

- Mettre les .tar des modèles dans un dossier nommé 'models'
- Mettre models dans le fichier principal
- Ouvrir le fichier python streamlit.py sur vscode
- Avec le terminal rendez vous dans le dossier ou est stocké streamlit.py
- Remplacer les chemins au début du code par les chemins de vos fichiers
- Lancer le site web avec 'streamlit run streamlit.py'
- Uploader une image pour faire le test (ATTENTION seuls les .jpg fonctionnent)

# Mise en place notre_modèle

- Ouvrir le fichier avec jupyter notebook
- Installer numpy / keras / tensorflow (ou tensorflowgpu si le programme est lancé sur un gpu)
- Remplacer les chemins d'accès par 
- Lancer le programme (Peut être très long à éxécuter en fonction de la machine)


# Conclusion

Notre projet avait pour but de mieux comprendre le monde de la reconnaissance d'image dans le domaine de la biodiversité. Nous avons pu réaliser un comparatif des différents modèles existants et nous avons même pu créer notre propre modèle. Cependant ce domaine est toujours en expansion et même les modèles les plus performants s'inclinent lorsque l'image est trop bruitée. 





