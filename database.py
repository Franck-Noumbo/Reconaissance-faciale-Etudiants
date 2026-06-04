import pickle
import os
from datetime import datetime
import streamlit as st
import cv2

# ==================== INITIALISATION DE SESSION ====================
def init_session_state():
    """Initialiser l'état de session"""
    if 'arcface_model' not in st.session_state:
        st.session_state.arcface_model = None
    if 'segmentation_model' not in st.session_state:
        st.session_state.segmentation_model = None
    if 'faces_db' not in st.session_state:
        st.session_state.faces_db = {}
    if 'face_embeddings' not in st.session_state:
        st.session_state.face_embeddings = {}
    if 'students_db' not in st.session_state:
        st.session_state.students_db = {}
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
    if 'captured_embedding' not in st.session_state:
        st.session_state.captured_embedding = None
    if 'recognition_result' not in st.session_state:
        st.session_state.recognition_result = None
    if 'use_white_background' not in st.session_state:
        st.session_state.use_white_background = True

# ==================== SAUVEGARDE ÉTUDIANT ====================
def sauvegarder_etudiant(image_bgr, student_data):
    """Sauvegarder un étudiant avec toutes ses informations"""
    try:
        from face_recognition import detecter_visage_deepface, extraire_features_arcface
        
        nom = student_data['nom'].strip().upper()
        prenom = student_data['prenom'].strip().title()
        age = int(student_data['age'])
        filiere = student_data['filiere']
        niveau = student_data['niveau']
        matricule = student_data['matricule'].strip()
        
        student_id = f"{matricule}_{nom}_{prenom}".replace(" ", "_")
        os.makedirs("etudiants_faces", exist_ok=True)
        
        face_resized, _ = detecter_visage_deepface(
            image_bgr,
            use_white_background=st.session_state.use_white_background,
            segmentation_model=st.session_state.segmentation_model
        )
        
        if face_resized is None:
            return False, "Aucun visage détecté"
        
        embedding_data = extraire_features_arcface(
            face_resized, 
            st.session_state.arcface_model
        )
        
        if embedding_data is None:
            return False, "Erreur extraction features"
        
        chemin_image = f"etudiants_faces/{student_id}.jpg"
        cv2.imwrite(chemin_image, face_resized)
        
        student_info = {
            'id': student_id,
            'nom': nom,
            'prenom': prenom,
            'full_name': f"{prenom} {nom}",
            'age': age,
            'filiere': filiere,
            'niveau': niveau,
            'matricule': matricule,
            'image_path': chemin_image,
            'date_enregistrement': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'embeddings_created': True
        }
        
        st.session_state.students_db[student_id] = student_info
        st.session_state.faces_db[student_id] = chemin_image
        st.session_state.face_embeddings[student_id] = embedding_data
        
        sauvegarder_database()
        
        return True, None
        
    except Exception as e:
        return False, str(e)

# ==================== SAUVEGARDER BASE DE DONNÉES ====================
def sauvegarder_database():
    """Sauvegarder la base de données complète"""
    try:
        data = {
            'students_db': st.session_state.students_db,
            'faces_db': st.session_state.faces_db,
            'face_embeddings': st.session_state.face_embeddings,
            'model_type': 'ArcFace ResNet100',
            'use_white_background': st.session_state.use_white_background,
            'total_students': len(st.session_state.students_db),
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('etudiants_database.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        backup_file = f"etudiants_database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(backup_file, 'wb') as f:
            pickle.dump(data, f)
            
        return True
    except Exception as e:
        st.error(f"❌ Erreur sauvegarde: {str(e)}")
        return False

# ==================== CHARGER BASE DE DONNÉES ====================
def charger_database():
    """Charger la base de données étudiants"""
    try:
        if os.path.exists('etudiants_database.pkl'):
            with open('etudiants_database.pkl', 'rb') as f:
                data = pickle.load(f)
                
                st.session_state.students_db = data.get('students_db', {})
                st.session_state.faces_db = data.get('faces_db', {})
                st.session_state.face_embeddings = data.get('face_embeddings', {})
                st.session_state.use_white_background = data.get('use_white_background', True)
                
                if st.session_state.students_db:
                    missing_in_embeddings = []
                    for student_id in st.session_state.students_db.keys():
                        if student_id not in st.session_state.face_embeddings:
                            missing_in_embeddings.append(student_id)
                    
                    if missing_in_embeddings:
                        st.warning(f"⚠️ {len(missing_in_embeddings)} étudiants sans embeddings détectés")
                    
                    st.success(f"✅ {len(st.session_state.students_db)} étudiants chargés avec succès!")
                    
    except Exception as e:
        st.warning(f"⚠️ Erreur chargement: {str(e)}")
        st.session_state.students_db = {}
        st.session_state.faces_db = {}
        st.session_state.face_embeddings = {}

# ==================== SUPPRIMER BASE DE DONNÉES ====================
def supprimer_base_donnees():
    """Supprimer complètement la base de données"""
    try:
        backup_file = f"etudiants_database_backup_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        data = {
            'students_db': st.session_state.students_db,
            'faces_db': st.session_state.faces_db,
            'face_embeddings': st.session_state.face_embeddings,
            'backup_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'reason': 'manual_deletion'
        }
        
        with open(backup_file, 'wb') as f:
            pickle.dump(data, f)
        
        for student_id, info in st.session_state.students_db.items():
            if 'image_path' in info and os.path.exists(info['image_path']):
                try:
                    os.remove(info['image_path'])
                except:
                    pass
        
        if os.path.exists('etudiants_database.pkl'):
            os.remove('etudiants_database.pkl')
        
        st.session_state.students_db = {}
        st.session_state.faces_db = {}
        st.session_state.face_embeddings = {}
        
        backup_files = sorted([f for f in os.listdir('.') if f.startswith('etudiants_database_backup')])
        for old_backup in backup_files[:-5]:
            try:
                os.remove(old_backup)
            except:
                pass
        
        return True, None
        
    except Exception as e:
        return False, str(e)

# ==================== SUPPRIMER UN ÉTUDIANT ====================
def supprimer_etudiant(student_id):
    """Supprimer un étudiant spécifique"""
    try:
        if student_id not in st.session_state.students_db:
            return False, "Étudiant non trouvé"
        
        student_info = st.session_state.students_db[student_id]
        os.makedirs("etudiants_deleted", exist_ok=True)
        backup_file = f"etudiants_deleted/{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        with open(backup_file, 'wb') as f:
            pickle.dump(student_info, f)
        
        if 'image_path' in student_info and os.path.exists(student_info['image_path']):
            try:
                os.remove(student_info['image_path'])
            except:
                pass
        
        if student_id in st.session_state.students_db:
            del st.session_state.students_db[student_id]
        if student_id in st.session_state.faces_db:
            del st.session_state.faces_db[student_id]
        if student_id in st.session_state.face_embeddings:
            del st.session_state.face_embeddings[student_id]
        
        sauvegarder_database()
        
        return True, None
        
    except Exception as e:
        return False, str(e)

# ==================== GESTION DU MODAL ====================
def afficher_modal_suppression():
    """Afficher le modal de confirmation pour suppression"""
    if st.session_state.modal_state.get('show', False):
        # Créer un conteneur pour le modal
        modal_container = st.empty()
        
        with modal_container.container():
            # Overlay semi-transparent
            st.markdown("""
            <style>
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.85);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 9999;
                padding: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Contenu du modal
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style='
                    background: linear-gradient(135deg, rgba(3, 19, 46, 0.95), rgba(8, 28, 64, 0.98));
                    padding: 2rem;
                    border-radius: 20px;
                    border: 1px solid rgba(239, 68, 68, 0.3);
                    box-shadow: 0 10px 40px rgba(239, 68, 68, 0.2);
                    text-align: center;
                '>
                """, unsafe_allow_html=True)
                
                st.markdown(f"<h2 style='color:#EF4444;'>⚠️ CONFIRMATION DE SUPPRESSION</h2>", unsafe_allow_html=True)
                
                if st.session_state.modal_state['target'] == 'all':
                    st.markdown("""
                    <p style='color:#A0B3C5;'>
                        Voulez-vous vraiment supprimer TOUTE la base de données ?<br><br>
                        Tous les étudiants, photos et données seront définitivement effacés.<br><br>
                        <strong style="color: #EF4444;">Cette action est irréversible !</strong>
                    </p>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <p style='color:#A0B3C5;'>
                        Voulez-vous vraiment supprimer cet étudiant ?<br><br>
                        Cette action est irréversible.<br><br>
                        <strong style="color: #EF4444;">Cette action est irréversible !</strong>
                    </p>
                    """, unsafe_allow_html=True)
                
                # Boutons du modal
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ OUI, Supprimer", 
                               type="primary", 
                               use_container_width=True,
                               key="modal_confirm"):
                        # Exécuter la suppression
                        if st.session_state.modal_state['target'] == 'all':
                            success, message = supprimer_base_donnees()
                            if success:
                                st.success("✅ Base de données supprimée avec succès!")
                                time.sleep(1)
                            else:
                                st.error(f"❌ Erreur: {message}")
                        else:
                            success, message = supprimer_etudiant(st.session_state.modal_state['target'])
                            if success:
                                st.success("✅ Étudiant supprimé avec succès!")
                                time.sleep(1)
                            else:
                                st.error(f"❌ Erreur: {message}")
                        
                        # Fermer le modal
                        st.session_state.modal_state = {'show': False, 'target': None, 'action': None}
                        modal_container.empty()
                        st.rerun()
                
                with col_btn2:
                    if st.button("❌ Annuler", 
                               type="secondary", 
                               use_container_width=True,
                               key="modal_cancel"):
                        # Fermer le modal sans action
                        st.session_state.modal_state = {'show': False, 'target': None, 'action': None}
                        modal_container.empty()
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)