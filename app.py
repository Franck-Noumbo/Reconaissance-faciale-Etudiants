import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import warnings
warnings.filterwarnings('ignore')

# Import des modules
from face_recognition import *
from database import *

# Configuration de la page
st.set_page_config(
    page_title="SecureFace ID SDIA4 - Système Étudiant",
    page_icon="🧑‍🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* -------------------------------------- */
    /* PARAMÈTRES GLOBALS & FOND (Adouci)     */
    /* -------------------------------------- */
    .main { 
        background: #010614; 
    }
    .stApp { 
        background: linear-gradient(150deg, #010614 10%, #03132e 100%);
        color: #A0B3C5; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    }
    
    /* -------------------------------------- */
    /* TITRE (Ombre adoucie)                   */
    /* -------------------------------------- */
    .header-title {
        text-align: center;
        font-size: 3.5rem; 
        font-weight: 900;
        background: linear-gradient(135deg, #3B82F6, #06B6D4, #4F46E5); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px; 
        text-shadow: 0 4px 20px rgba(59, 130, 246, 0.25); 
        margin-bottom: 1rem;
    }

    /* -------------------------------------- */
    /* CARTES (Maintien de l'effet verre)     */
    /* -------------------------------------- */
    .face-card {
        background: rgba(3, 19, 46, 0.7); 
        backdrop-filter: blur(15px);
        border-radius: 20px; 
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(49, 140, 255, 0.3);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }

    /* -------------------------------------- */
    /* BOXES DE STATUT (Plus de profondeur)    */
    /* -------------------------------------- */
    .success-box {
        background: linear-gradient(135deg, #06B6D4, #14B8A6); 
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: #010614; 
        font-weight: 900;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3); 
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, #EF4444, #DC2626); 
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-weight: 900;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #F59E0B, #D97706); 
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: #010614; 
        font-weight: 900;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #3B82F6, #8B5CF6); 
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-weight: 900;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* -------------------------------------- */
    /* FORMULAIRES                            */
    /* -------------------------------------- */
    .student-form {
        background: rgba(3, 19, 46, 0.6);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin: 15px 0;
    }
    
    .form-title {
        color: #3B82F6;
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 15px;
        text-align: center;
    }
    
    /* -------------------------------------- */
    /* STATISTIQUES & BADGES                  */
    /* -------------------------------------- */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px; 
        margin-top: 20px;
    }
    .stat-item {
        background: rgba(3, 19, 46, 0.5); 
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease-in-out; 
    }
    .stat-item:hover {
        transform: translateY(-3px); 
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.2);
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 900;
        color: #3B82F6;
        margin-bottom: 5px;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #93C5FD;
    }
    
    .model-badge {
        display: inline-block;
        background: linear-gradient(90deg, #3B82F6, #06B6D4);
        color: white;
        padding: 5px 14px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 3px;
        text-transform: uppercase;
    }
    
    .hybrid-badge {
        background: linear-gradient(135deg, #10B981, #06B6D4); 
        color: #010614; 
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 1rem;
        font-weight: 800;
        display: inline-block;
        margin: 8px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    
    /* -------------------------------------- */
    /* PROFIL CARDS                           */
    /* -------------------------------------- */
    .profile-card {
        background: linear-gradient(135deg, rgba(3, 19, 46, 0.8), rgba(8, 28, 64, 0.9));
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
    }
    
    .profile-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
        transform: translateY(-3px);
    }
    
    /* -------------------------------------- */
    /* FICHE ÉTUDIANT                         */
    /* -------------------------------------- */
    .student-info-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(6, 182, 212, 0.1));
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin: 15px 0;
    }
    
    .student-name {
        font-size: 1.6rem;
        font-weight: 900;
        color: #3B82F6;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .student-details {
        background: rgba(255, 255, 255, 0.05);
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALISATION DE SESSION ====================
init_session_state()

# Variables pour la gestion du modal
if 'modal_state' not in st.session_state:
    st.session_state.modal_state = {
        'show': False,
        'target': None,
        'action': None
    }

# ==================== CHARGEMENT DES MODÈLES ====================
if st.session_state.arcface_model is None:
    with st.spinner("⚡ Chargement du modèle ArcFace ResNet100..."):
        arcface_model = load_arcface_model()
        if arcface_model:
            st.session_state.arcface_model = arcface_model
            st.success("✅ ArcFace ResNet100 chargé avec succès!")

if st.session_state.segmentation_model is None:
    with st.spinner("🎭 Chargement MediaPipe pour fond blanc..."):
        seg_model = load_mediapipe()
        st.session_state.segmentation_model = seg_model

# Charger la base
charger_database()

# ==================== INTERFACE ====================

# Header
st.markdown("<div class='header-title'>🧑‍🎓 Système de Reconnaissance Étudiant</div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#94a3b8;margin-bottom:2rem;font-size:1.1rem;'>
<span class='hybrid-badge'>ArcFace ResNet100 • GESTION ÉTUDIANTS</span>
</div>
""", unsafe_allow_html=True)

# Afficher le modal si nécessaire
afficher_modal_suppression()

# Toggle pour fond blanc
st.session_state.use_white_background = st.toggle(
    "⚪ Activer le fond blanc pour les photos",
    value=st.session_state.use_white_background,
    help="MediaPipe isolera le visage sur fond blanc pour les enregistrements"
)

# Statistiques
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    bg_status = "✅ ON" if st.session_state.use_white_background else "❌ OFF"
    st.markdown(f"<div style='font-size:1.3rem;font-weight:800;color:#8b5cf6;'>{bg_status}</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#94a3b8;font-size:0.9rem;'>Fond blanc</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div style='font-size:1.3rem;font-weight:800;color:#3b82f6;'>{len(st.session_state.students_db)}</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#94a3b8;font-size:0.9rem;'>Étudiants</div>", unsafe_allow_html=True)

with col3:
    total_embeddings = len(st.session_state.face_embeddings)
    st.markdown(f"<div style='font-size:1.3rem;font-weight:800;color:#10b981;'>{total_embeddings}</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#94a3b8;font-size:0.9rem;'>Embeddings</div>", unsafe_allow_html=True)

with col4:
    status = "🟢" if st.session_state.arcface_model else "🔴"
    st.markdown(f"<div style='font-size:1.3rem;font-weight:800;color:#f59e0b;'>{status}</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#94a3b8;font-size:0.9rem;'>ArcFace</div>", unsafe_allow_html=True)

with col5:
    mediapipe_status = "✅" if st.session_state.segmentation_model else "❌"
    st.markdown(f"<div style='font-size:1.3rem;font-weight:800;color:#ec4899;'>{mediapipe_status}</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#94a3b8;font-size:0.9rem;'>MediaPipe</div>", unsafe_allow_html=True)

with col6:
    if st.button("🗑️ Tout supprimer", type="secondary", use_container_width=True, key="delete_all_stats"):
        st.session_state.modal_state = {'show': True, 'target': 'all', 'action': None}
        st.rerun()

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# Interface principale
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("<div class='face-card'>", unsafe_allow_html=True)
    st.markdown("### 📸 **CAPTURE ÉTUDIANT**")
    
    camera_file = st.camera_input("📷 Prendre une photo de l'étudiant", key="student_camera")
    
    if camera_file is not None:
        image = Image.open(camera_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        st.session_state.captured_image = image_cv
        
        face_resized, face_rect = detecter_visage_deepface(
            image_cv,
            use_white_background=st.session_state.use_white_background,
            segmentation_model=st.session_state.segmentation_model
        )
        
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            display_img = image_cv.copy()
            if face_rect:
                x, y, w, h = face_rect
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (139, 92, 246), 4)
                bg_text = "FOND BLANC" if st.session_state.use_white_background else "ORIGINAL"
                cv2.putText(display_img, f"DEEPFACE • {bg_text}", (x, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 92, 246), 2)
            
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
                    caption="📸 Photo originale avec détection",
                    use_container_width=True)
        
        with img_col2:
            if face_resized is not None:
                st.image(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB),
                        caption="🎯 Visage prêt pour enregistrement" + 
                                (" (Fond blanc)" if st.session_state.use_white_background else ""),
                        use_container_width=True)
        
        if face_resized is not None:
            st.success("✅ Photo prête pour enregistrement étudiant!")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='face-card'>", unsafe_allow_html=True)
    st.markdown("### 📁 **IMPORTER UNE PHOTO DEPUIS UN FICHIER**")
    
    uploaded_file = st.file_uploader(
        "Choisir une photo d'étudiant depuis votre ordinateur", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Formats supportés: JPG, JPEG, PNG, BMP, WEBP"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        st.markdown("#### 🖼️ Image importée")
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.image(image, caption="Image originale", use_container_width=True)
        
        with st.spinner("🔍 Détection de visage en cours..."):
            face_resized, face_rect = detecter_visage_deepface(
                image_cv,
                use_white_background=st.session_state.use_white_background,
                segmentation_model=st.session_state.segmentation_model
            )
        
        with col_img2:
            if face_resized is not None:
                st.image(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB),
                        caption="Visage détecté" + 
                                (" (Fond blanc)" if st.session_state.use_white_background else ""),
                        use_container_width=True)
                st.session_state.captured_image = image_cv
                st.success("✅ Photo importée prête pour enregistrement étudiant!")
                
                if face_rect:
                    display_img = image_cv.copy()
                    x, y, w, h = face_rect
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), (59, 130, 246), 3)
                    cv2.putText(display_img, "VISAGE DETECTE", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (59, 130, 246), 2)
                    
                    st.markdown("#### 🔍 Zone de détection")
                    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
                            caption="Zone du visage détectée",
                            use_container_width=True)
            else:
                st.error("❌ Aucun visage détecté dans l'image")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='face-card'>", unsafe_allow_html=True)
    st.markdown("### ⚡ **IDENTIFICATION**")
    
    if st.button("🔍 IDENTIFIER ÉTUDIANT", 
                use_container_width=True, 
                type="primary",
                disabled=st.session_state.captured_image is None):
        with st.spinner("🧠 Identification en cours..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            student_id, confiance, student_info = identifier_visage_arcface(
                st.session_state.captured_image, 
                seuil=82.0
            )
            st.session_state.recognition_result = (student_id, confiance, student_info)
    
    if st.session_state.recognition_result:
        student_id, confiance, student_info = st.session_state.recognition_result
        
        if student_id == "INCONNU":    
                st.markdown(f"""<div class='error-box'>
                        ❌ <strong>ÉTUDIANT INCONNU</strong><br>
                    </div>""", unsafe_allow_html=True)
        
        elif student_id in ["BASE_DE_DONNEES_VIDE", "AUCUN_VISAGE", "ERREUR_EXTRACTION"]:
            messages = {
                "BASE_DE_DONNEES_VIDE": "La base de données est vide",
                "AUCUN_VISAGE": "Aucun visage détecté dans l'image",
                "ERREUR_EXTRACTION": "Erreur lors de l'extraction des features"
            }
            st.markdown(f"<div class='warning-box'>⚠️ {messages.get(student_id, student_id)}</div>", unsafe_allow_html=True)
        
        else:
            if student_info:
                html_content = afficher_infos_etudiant(student_info, confiance)
                if html_content:
                    st.markdown(html_content, unsafe_allow_html=True)
                    
                    if 'image_path' in student_info and os.path.exists(student_info['image_path']):
                        try:
                            st.markdown("---")
                            st.markdown("### 📷 **Photo enregistrée**")
                            img = cv2.imread(student_info['image_path'])
                            if img is not None:
                                nom_photo = student_info.get('full_name', 'étudiant')
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                        caption=f"Photo de {nom_photo}",
                                        use_container_width=True)
                        except:
                            pass
            else:
                st.markdown(f"""<div class='info-box'>
                    🎯 <strong>ÉTUDIANT IDENTIFIÉ</strong><br>
                    <span style='font-size:1.4em;font-weight:900;'>{student_id}</span><br>
                    <span style='color:#3b82f6;font-size:1.2em;font-weight:800;'>Confiance: {confiance:.1f}%</span><br>
                    <small>Informations détaillées non disponibles</small>
                </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📝 **FORMULAIRE ÉTUDIANT**")
    
    with st.form(key='student_form'):
        st.markdown("<div class='student-form'>", unsafe_allow_html=True)
        st.markdown("<div class='form-title'>📋 Informations de l'étudiant</div>", unsafe_allow_html=True)
        
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            nom = st.text_input("Nom de famille", placeholder="NOUMBO", key='nom_input')
            prenom = st.text_input("Prénom", placeholder="EDWIN", key='prenom_input')
            age = st.number_input("Âge", min_value=16, max_value=70, value=20, key='age_input')
        
        with col_form2:
            matricule = st.text_input("Matricule", placeholder="22G00604", key='matricule_input')
            filieres = [
                "SDIA4", "SDIA3", "MEMA4", "MEMA5", 
                "MEMA3", "Génie Civil", "Génie Électrique", "Génie Mécanique",
                "SDIA5", "Gestion", "Droit", "Médecine", "Pharmacie",
                "Lettres", "Sciences Sociales", "Arts", "Autre"
            ]
            filiere = st.selectbox("Filière", filieres, key='filiere_input')
            
            niveaux = [
                "Licence 1", "Licence 2", "Licence 3", 
                "Master 1", "Master 2", "Doctorat",
                "Professeur", "Administratif"
            ]
            niveau = st.selectbox("Niveau", niveaux, key='niveau_input')
        
        submitted = st.form_submit_button("💾 ENREGISTRER L'ÉTUDIANT", 
                                         use_container_width=True,
                                         type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submitted:
            if not nom or not prenom or not matricule:
                st.error("❌ Veuillez remplir tous les champs obligatoires")
            elif st.session_state.captured_image is None:
                st.error("❌ Aucune photo capturée. Veuillez prendre une photo d'abord.")
            else:
                student_data = {
                    'nom': nom,
                    'prenom': prenom,
                    'age': age,
                    'filiere': filiere,
                    'niveau': niveau,
                    'matricule': matricule
                }
                
                with st.spinner("💾 Enregistrement en cours..."):
                    success, message = sauvegarder_etudiant(st.session_state.captured_image, student_data)
                    
                    if success:
                        st.success("✅ Étudiant enregistré avec succès!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Erreur: {message}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== AFFICHAGE DES ÉTUDIANTS ====================
if st.session_state.students_db:
    st.markdown("---")
    st.markdown("### 👥 **BASE DE DONNÉES ÉTUDIANTS**")
    
    col_save, col_delete = st.columns(2)
    with col_save:
        if st.button("💾 Sauvegarder la base", use_container_width=True, key="save_db"):
            if sauvegarder_database():
                st.success("✅ Base de données sauvegardée!")
                time.sleep(1)
            else:
                st.error("❌ Erreur lors de la sauvegarde")
    
    with col_delete:
        if st.button("🗑️ Supprimer toute la base", use_container_width=True, type="secondary", key="delete_all_db"):
            st.session_state.modal_state = {'show': True, 'target': 'all', 'action': None}
            st.rerun()
    
    students = list(st.session_state.students_db.values())
    students.sort(key=lambda x: x.get('nom', ''))
    
    for i in range(0, len(students), 3):
        cols = st.columns(3)
        for idx, student in enumerate(students[i:i+3]):
            with cols[idx]:
                try:
                    student_id = student.get('id', '')
                    image_path = student.get('image_path', '')
                    
                    if os.path.exists(image_path):
                        img = cv2.imread(image_path)
                        if img is not None:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                    caption=f"**{student.get('prenom', '')} {student.get('nom', '')}**",
                                    use_container_width=True)
                            
                            st.markdown(f"""
                            <div class='profile-card'>
                                <strong>🎓 {student.get('prenom', '')} {student.get('nom', '')}</strong><br>
                                <small>• <strong>Âge:</strong> {student.get('age', 'N/A')} ans</small><br>
                                <small>• <strong>Matricule:</strong> {student.get('matricule', 'N/A')}</small><br>
                                <small>• <strong>Filière:</strong> {student.get('filiere', 'N/A')}</small><br>
                                <small>• <strong>Niveau:</strong> {student.get('niveau', 'N/A')}</small><br>
                                <small>• <strong>Date:</strong> {student.get('date_enregistrement', 'N/A')}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button("🗑️ Supprimer", 
                                       key=f"delete_{student_id}",
                                       use_container_width=True):
                                st.session_state.modal_state = {'show': True, 'target': student_id, 'action': None}
                                st.rerun()
                except Exception as e:
                    st.error(f"Erreur chargement: {str(e)}")

# ==================== PIED DE PAGE ====================
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1.5rem;background:linear-gradient(135deg,rgba(30,41,59,0.8),rgba(15,23,42,0.9));border-radius:15px;border:1px solid rgba(139,92,246,0.3);'>
    <div style='font-size:1.2rem;font-weight:800;margin-bottom:10px;background:linear-gradient(135deg,#8b5cf6,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🧑‍🎓 Système de Reconnaissance Étudiant SDIA4
    </div>
    <div style='color:#94a3b8;font-size:0.95rem;'>
        • ArcFace ResNet100 • Gestion Étudiants • Formulaire Complet avec Âge • 
        Base de Données Structurée • Précision 99%+ • Affichage Dynamique
    </div>
    <div style='color:#64748b;font-size:0.8rem;margin-top:10px;'>
        🔒 Toutes les suppressions sont précédées d'une sauvegarde automatique
    </div>
</div>
""", unsafe_allow_html=True)