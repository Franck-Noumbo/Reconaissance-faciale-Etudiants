# Projet Reconnaissance Faciale Étudiants

Système intelligent de reconnaissance faciale développé pour l’identification des étudiants à l’aide de l’intelligence artificielle et des réseaux de neurones convolutionnels (CNN).

## 📌 Description du projet

Ce projet a été conçu dans le cadre d’un système d’identification automatique des étudiants. Il utilise la reconnaissance faciale basée sur le modèle **ArcFace** afin de détecter et reconnaître les visages de manière rapide et sécurisée.

L’objectif principal est d’améliorer l’identification des étudiants dans un environnement académique.

## ✨ Fonctionnalités

- 🔍 Détection et reconnaissance faciale
- 👨‍🎓 Identification automatique des étudiants
- 🗄️ Gestion des données des étudiants
- ⚡ Traitement rapide des visages
- 🖥️ Interface simple d’utilisation

## 📋 Prérequis

Avant d’exécuter le projet, assurez-vous d’avoir :

- Windows 10/11 ou Linux
- Python 3.9+
- Minimum 8 Go de RAM
- GPU NVIDIA recommandé (optionnel)

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Franck-Noumbo/projets-reconnaissance-faciale-etudiants.git
cd projets-reconnaissance-faciale-etudiants
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger le modèle ArcFace

Le fichier `arcfaceresnet100.onnx` n’est pas inclus dans ce dépôt à cause des limitations de taille GitHub.

Téléchargez le modèle ici :

:contentReference[oaicite:0]{index=0}

Puis placez le fichier :

```text
arcfaceresnet100.onnx
```

dans le dossier principal du projet.

### 4. Lancer l’application

#### Option 1 : Windows (recommandé)

```bash
run_app.bat
```

#### Option 2 : VS Code / Terminal Python

```bash
python run_app.py
```

## 📂 Structure du projet

```text
Projet-Reconnaissance-Faciale-Etudiants/
│── app.py
│── database.py
│── face_recognition.py
│── run_app.py
│── run_app.bat
│── requirements.txt
│── arcfaceresnet100.onnx
```

## ⚠️ Remarque

Le modèle ONNX n’est pas fourni directement dans ce dépôt GitHub afin de respecter les limitations de taille des fichiers.

## 👨‍💻 Auteur

**Franck Noumbo**  
Projet : **Projets Reconnaissance Faciale Étudiants**