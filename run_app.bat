@echo off
echo ========================================
echo  SecureFace ID SDIA4 - ENPD
echo  Systeme de Reconnaissance Faciale
echo ========================================
echo.

REM Créer les dossiers nécessaires
if not exist "data" mkdir data
if not exist "data\etudiants_faces" mkdir data\etudiants_faces
if not exist "data\backups" mkdir data\backups
if not exist "data\deleted" mkdir data\deleted

REM Vérifier le modèle ArcFace
if not exist "arcfaceresnet100.onnx" (
    echo ⚠️ ATTENTION: Modele ArcFace non trouve!
    echo Téléchargez le depuis: https://github.com/deepinsight/insightface
    echo Ou utilisez DeepFace sans modele ONNX.
    timeout /t 5
)

REM Configurer l'environnement GPU
set CUDA_VISIBLE_DEVICES=0
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_CPP_MIN_LOG_LEVEL=2

REM Lancer l'application
echo.
echo 🚀 Lancement de SecureFace ID...
echo 📍 Application disponible sur: http://localhost:8501
echo 📍 Appuyez sur Ctrl+C pour arreter
echo.

streamlit run app.py --server.port 8501 --server.address 0.0.0.0

pause