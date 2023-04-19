import streamlit as st
from PIL import Image
from torchvision.models import vgg11
from torchvision.models import resnet18
from torchvision.models import resnet101
import os
from PIL import Image, ImageOps, ImageEnhance
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import json
import random
import numpy as np


# Chemins d'accès aux fichiers
json_path = "plantnet300K_species_names.json"
image_test_path = 'C:/Users/cleth/Desktop/Fichier_projet_E4/plantnet_300K/images_test'
model_vgg11_path = 'models/vgg11_weights_best_acc.tar'
model_resnet18_path = 'models/resnet18_weights_best_acc.tar'
model_resnet101_path = 'models/resnet101_weights_best_acc.tar'


st.title("Classification d'espèce")

# Bouton upload image
image_file = st.file_uploader(
    "Choisissez une image", type=['jpeg', 'jpg'])

# Liste déroulante sélection de modèle
type_espece = st.selectbox("Choisir le modèle", [
                           'vgg11', 'resnet18', 'resnet101'])

# Une fois qu'une image est rentrée
if image_file is not None:

    # On récupére l'image
    image = Image.open(image_file)
    st.subheader("Prédiction du modèle")

    # On créee des fonctions qu'on pourra appeller ensuite

    # Permet de retrouver le nom de la plante à partir de son numéro
    def trouver_nom_plante(numero, data):
        if str(numero) in data:
            return data[str(numero)]
        else:
            return "Numéro introuvable"

    def nom_plante(df, file_names):
        with open(json_path, "r") as f:
            data = json.load(f)

        images = []

        # On boucle sur les 5 espèces les plus probables et on affiche une photo de chaque espèce prise au hasard dans le fichier image_test
        for i in range(len(df)):
            # Charger l'image correspondant au fichier numero_plante
            numero_plante = file_names[df['indice'][i]]

            nom_fichier = image_test_path + '/' + numero_plante
            name = os.listdir(nom_fichier)
            random_file_name = random.choice(name)
            img_path = nom_fichier + "/" + random_file_name
            img = Image.open(img_path)
            img = img.resize((200, 200))
            images.append(img)

            # Trouver le nom de la plante
            df.loc[i, 'Nom'] = trouver_nom_plante(numero_plante, data)

        return (df, images)

    # Permet de récupérer un modèle préentrainer et de le charger
    def loadmodel(model, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError
        d = torch.load(filename, map_location=torch.device('cpu'))
        model.load_state_dict(d['model'])
        return d['epoch']

    # Fonction principale qui va gérer la sélection de modèle ainsi que la prédiction d'espèce
    def plante(img):

        if (type_espece == 'vgg11'):
            model = vgg11(num_classes=1081)
            model_path = model_vgg11_path
        elif (type_espece == 'resnet18'):
            model = resnet18(num_classes=1081)
            model_path = model_resnet18_path
        else:
            model = resnet101(num_classes=1081)
            model_path = model_resnet101_path

        loadmodel(model, model_path)

        # Permet de récupérer les noms des fichiers
        file_names = os.listdir(image_test_path)

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        x = transform(img)
        x = x.unsqueeze(0)

        model.eval()
        output = model(x)

        pred = top_values, top_indices = torch.topk(output, k=5)

        probs = F.softmax(output, dim=1).detach()
        top_probs = probs[0, top_indices[0]]

        values = pred.values.squeeze().tolist()
        indices = pred.indices.squeeze().tolist()
        proba = top_probs.squeeze().tolist()

        data_dict = {'indice': indices, 'values': values, 'proba': proba}
        df = pd.DataFrame(data_dict)
        df["Nom"] = 'inconnu'

        result = nom_plante(df, file_names)

        return result

    # On exécute notre code sur l'image récupérée
    result = plante(image)

    # Permet de créer bandeau avec les 5 images des espèces les plus probables
    images_np = [np.asarray(img) for img in result[1]]
    height, width = images_np[0].shape[:2]
    new_width = 200
    new_height = int(new_width * height / width)
    concatenated_image = np.concatenate(images_np, axis=1)
    final_image = Image.fromarray(concatenated_image)
    final_image = final_image.resize((len(result[1]) * new_width, new_height))

    # Afficher l'image
    st.image(image, caption='Image originale', width=250)
    st.image(final_image, caption='5 espèces possibles', use_column_width=True)

    tableau = result[0].drop(columns={'indice', 'values'})
    st.dataframe(tableau)
