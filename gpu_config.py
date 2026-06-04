"""
Configuration GPU pour SecureFace ID SDIA4
Force l'utilisation du GPU NVIDIA
"""

import os
import sys

def force_gpu_usage():
    """
    Force l'utilisation du GPU NVIDIA
    Doit être appelé AVANT tout import de tensorflow/pytorch
    """
    print("⚡ Configuration GPU forcée...")
    
    # 1. Forcer CUDA GPU 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 2. Permettre la croissance mémoire (évite OOM)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # 3. Réduire les logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=tous, 1=info, 2=warnings, 3=erreurs
    
    # 4. Optimiser pour Quadro P620
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    
    print("✅ Configuration GPU appliquée")

def check_gpu_status():
    """
    Vérifie l'état du GPU et affiche les informations
    """
    import subprocess
    
    print("\n" + "="*50)
    print("DIAGNOSTIC SYSTÈME")
    print("="*50)
    
    # Vérifier nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split(',')
            print(f"🎮 GPU Détecté: {gpu_info[0].strip()}")
            print(f"📊 Mémoire Totale: {gpu_info[1].strip()} MB")
            print(f"📊 Mémoire Libre: {gpu_info[2].strip()} MB")
            print(f"🔧 Driver: {gpu_info[3].strip()}")
        else:
            print("⚠️ NVIDIA GPU non détecté (nvidia-smi échoue)")
    except Exception as e:
        print(f"⚠️ Impossible d'exécuter nvidia-smi: {e}")
    
    # Vérifier TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow voit {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("❌ TensorFlow ne voit aucun GPU")
    except ImportError:
        print("ℹ️ TensorFlow non installé")
    except Exception as e:
        print(f"⚠️ Erreur TensorFlow: {e}")
    
    print("="*50 + "\n")

# Exécuter automatiquement au chargement
if __name__ != "__main__":
    force_gpu_usage()
    check_gpu_status()