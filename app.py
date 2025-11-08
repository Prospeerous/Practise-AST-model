from sqlalchemy import func
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import librosa.display

from database.models import db, User, Patient, AudioFile, AnalysisResult

import numpy as np
import json

# Two-Factor Authentication imports
import pyotp
import qrcode
import io
import base64

# ==================== CNN-LSTM Model Imports ====================
import tensorflow as tf
from tensorflow import keras
import librosa
import traceback
# ===========================================================================

# ==================== Load CNN-LSTM Model ====================
print("Loading CNN-LSTM model...")

try:
    # Load model without compilation to fix batch_shape error
    cnn_lstm_model = keras.models.load_model(
        "outputs/cnn_lstm_model/best_cnn_lstm_model.h5",
        compile=False
    )
    
    # Manually compile the model
    cnn_lstm_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("[OK] Model loaded successfully")
    print(f"Model input shape: {cnn_lstm_model.input_shape}")
    print(f"Model output shape: {cnn_lstm_model.output_shape}")

except Exception as e:
    print(f"[ERROR] Error loading model: {str(e)}")
    traceback.print_exc()
    exit(1)

# Load class mappings and normalization parameters
with open("outputs/cnn_lstm_model/class_mapping_and_weights.json", "r") as f:
    class_mapping_data = json.load(f)
print("[OK] Class mappings loaded")

with open("outputs/cnn_lstm_model/normalization_params.json", "r") as f:
    norm_params = json.load(f)
print("[OK] Normalization parameters loaded")

# Extract disease labels - THIS IS THE ONLY OUTPUT
DISEASE_LABELS = ['Healthy', 'Asthma', 'Bronchitis', 'Pneumonia', 'COPD']
print(f"[OK] Disease classes: {DISEASE_LABELS}")

# Mel spectrogram parameters
with open("outputs/cnn_lstm_model/cell1_configuration.json", "r") as f:
    cell_config = json.load(f)
print("[OK] Cell configuration loaded")

# Extract mel spectrogram settings
MEL_PARAMS = {
    'n_mels': cell_config.get('n_mels', 128),
    'n_fft': cell_config.get('n_fft', 2048),
    'hop_length': cell_config.get('hop_length', 512),
    'target_sr': cell_config.get('sample_rate', 16000),
    'clip_duration': cell_config.get('clip_duration', 5)
}

print(f"[OK] CNN-LSTM Model ready with {len(DISEASE_LABELS)} disease classes")
print(f"[OK] Mel parameters: {MEL_PARAMS}")

# ==================== MODEL HEALTH DIAGNOSIS ====================
print("\n" + "="*70)
print("DIAGNOSING MODEL HEALTH - CHECKING FOR COLLAPSE")
print("="*70)

# Test 1: Random inputs - should give VARIED predictions if model is healthy
print("\nTest 1: Predictions on 10 random inputs")
print("-" * 70)
random_predictions = []
for i in range(10):
    # Create random input matching model's expected shape
    random_input = np.random.randn(1, 157, 128, 1).astype(np.float32)
    pred = cnn_lstm_model.predict(random_input, verbose=0)
    pred_class = np.argmax(pred[0])
    pred_conf = pred[0][pred_class] * 100
    random_predictions.append(pred_class)
    
    print(f"  Random Test {i+1:2d}: {DISEASE_LABELS[pred_class]:12s} ({pred_conf:5.1f}%) | Probs: {pred[0]}")

# Analyze prediction diversity
unique_predictions = len(set(random_predictions))
most_common = max(set(random_predictions), key=random_predictions.count)
most_common_count = random_predictions.count(most_common)

print("\n" + "-" * 70)
print(f"Diversity Analysis:")
print(f"   Unique classes predicted: {unique_predictions} out of {len(DISEASE_LABELS)}")
print(f"   Most common prediction: {DISEASE_LABELS[most_common]} ({most_common_count}/10 times)")

if unique_predictions == 1:
    print("\n[WARNING] CRITICAL WARNING: Model is COLLAPSED!")
    print("   [X] Model predicts ONLY ONE CLASS for all inputs")
    print("   [X] This model CANNOT be fixed without retraining")
    print("   [X] Predictions will be UNRELIABLE")
elif unique_predictions <= 2:
    print("\n[WARNING] WARNING: Model has LIMITED diversity")
    print("   [!] Model may be biased toward certain classes")
    print("   [!] Retraining recommended for better performance")
elif most_common_count >= 7:
    print("\n[WARNING] WARNING: Model shows STRONG BIAS")
    print(f"   [!] Predicts {DISEASE_LABELS[most_common]} in {most_common_count*10}% of cases")
    print("   [!] Retraining recommended with better class balancing")
else:
    print("\n[OK] Model appears healthy with good prediction diversity")

# Test 2: Check model layer weights for degeneration
print("\nTest 2: Model Weight Statistics (first 5 trainable layers)")
print("-" * 70)
layer_count = 0
for i, layer in enumerate(cnn_lstm_model.layers):
    if len(layer.get_weights()) > 0 and layer_count < 5:
        weights = layer.get_weights()[0]
        weight_mean = weights.mean()
        weight_std = weights.std()
        weight_min = weights.min()
        weight_max = weights.max()
        
        print(f"  Layer {i:2d} ({layer.name:20s}): ", end="")
        print(f"mean={weight_mean:7.4f}, std={weight_std:7.4f}, ", end="")
        print(f"range=[{weight_min:7.4f}, {weight_max:7.4f}]")
        
        layer_count += 1

print("\n" + "="*70)
print("Diagnosis complete. Starting Flask application...")
print("="*70 + "\n")
# ================================================================

# ==================== CNN-LSTM Prediction Functions ====================
def preprocess_audio_for_cnn_lstm(filepath):
    """
    Preprocess audio file to mel spectrogram for CNN-LSTM model
    Returns normalized mel spectrogram or None if processing fails
    """
    try:
        print(f"\n=== Starting audio preprocessing for: {os.path.basename(filepath)} ===")
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"ERROR: File does not exist: {filepath}")
            return None
        
        # Load audio file
        print(f"Loading audio with sr={MEL_PARAMS['target_sr']}...")
        audio, sr = librosa.load(filepath, sr=MEL_PARAMS['target_sr'], mono=True)
        
        # Validate audio loaded successfully
        if audio is None or len(audio) == 0:
            print("ERROR: Audio file is empty or couldn't be loaded")
            return None
        
        print(f"[OK] Audio loaded: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f} seconds)")
        
        # Trim or pad to fixed duration
        target_length = MEL_PARAMS['clip_duration'] * MEL_PARAMS['target_sr']
        if len(audio) > target_length:
            audio = audio[:target_length]
            print(f"[OK] Truncated audio to {MEL_PARAMS['clip_duration']} seconds")
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            print(f"[OK] Padded audio to {MEL_PARAMS['clip_duration']} seconds")
        
        # Generate mel spectrogram
        print(f"Generating mel spectrogram (n_mels={MEL_PARAMS['n_mels']}, n_fft={MEL_PARAMS['n_fft']})...")
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=MEL_PARAMS['n_mels'],
            n_fft=MEL_PARAMS['n_fft'],
            hop_length=MEL_PARAMS['hop_length'],
            fmin=50,
            fmax=8000
        )
        
        # Validate spectrogram
        if mel_spec.size == 0 or mel_spec.shape[0] == 0 or mel_spec.shape[1] == 0:
            print(f"ERROR: Mel spectrogram is empty. Shape: {mel_spec.shape}")
            return None
        
        print(f"[OK] Mel spectrogram generated: {mel_spec.shape}")
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Check for NaN or Inf values
        if np.isnan(mel_spec_db).any() or np.isinf(mel_spec_db).any():
            print("WARNING: Mel spectrogram contains NaN or Inf values, replacing with 0")
            mel_spec_db = np.nan_to_num(mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Mel spec dB range: [{mel_spec_db.min():.2f}, {mel_spec_db.max():.2f}]")
        
        # ============================================================
        # CRITICAL: Use EXACT normalization values from training
        # ============================================================
        mean = -65.2695  # EXACT value from training Cell 4
        std = 17.8257    # EXACT value from training Cell 4
        
        print(f"Normalizing with EXACT training values: mean={mean:.4f}, std={std:.4f}...")
        mel_spec_normalized = (mel_spec_db - mean) / std
        
        # Validate normalized spectrogram
        if np.isnan(mel_spec_normalized).any() or np.isinf(mel_spec_normalized).any():
            print("WARNING: Normalized spectrogram contains NaN or Inf, replacing with 0")
            mel_spec_normalized = np.nan_to_num(mel_spec_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Normalized spec range: [{mel_spec_normalized.min():.2f}, {mel_spec_normalized.max():.2f}]")
        
        # Reshape for model input: (batch, height, width, channels)
        mel_spec_normalized = mel_spec_normalized.T  # Transpose to (time, freq)
        mel_spec_normalized = np.expand_dims(mel_spec_normalized, axis=0)  # Add batch dimension
        mel_spec_normalized = np.expand_dims(mel_spec_normalized, axis=-1)  # Add channel dimension
        
        print(f"[OK] Final feature shape for model: {mel_spec_normalized.shape}")
        print(f"Expected model input shape: {cnn_lstm_model.input_shape}")
        print("=== Audio preprocessing completed successfully ===\n")
        
        return mel_spec_normalized
        
    except Exception as e:
        print(f"ERROR in audio preprocessing: {str(e)}")
        traceback.print_exc()
        return None

def predict_disease_diagnosis(filepath):
    """
    Predict respiratory disease diagnosis using CNN-LSTM model
    Returns: dict with prediction results or error
    """
    try:
        print(f"\n{'='*60}")
        print(f"Starting disease diagnosis for: {os.path.basename(filepath)}")
        print(f"{'='*60}")
        
        # Preprocess audio
        mel_spec = preprocess_audio_for_cnn_lstm(filepath)
        
        # Check if preprocessing failed
        if mel_spec is None:
            error_msg = "Failed to extract features from audio. Audio file may be corrupted or in an unsupported format."
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'disease_label': 'Unknown',
                'disease_confidence': 0.0
            }
        
        # Validate input is not empty
        if mel_spec.size == 0:
            error_msg = "Mel spectrogram is empty after preprocessing"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'disease_label': 'Unknown',
                'disease_confidence': 0.0
            }
        
        print("\nMaking prediction with CNN-LSTM model...")
        
        # Make prediction
        predictions = cnn_lstm_model.predict(mel_spec, verbose=0)
        
        print(f"[OK] Prediction completed")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction values: {predictions[0]}")
        
        # Validate predictions
        if predictions is None or predictions.size == 0:
            error_msg = "Model returned empty prediction"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'disease_label': 'Unknown',
                'disease_confidence': 0.0
            }
        
        # Get disease probabilities (single output)
        disease_probs = predictions[0]
        
        # Validate probability array
        if disease_probs.size == 0:
            error_msg = "Disease prediction probabilities array is empty"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'disease_label': 'Unknown',
                'disease_confidence': 0.0
            }
        
        # Get disease prediction
        print("\nProcessing disease prediction...")
        print(f"Disease probabilities: {disease_probs}")
        
        disease_idx = np.argmax(disease_probs)
        disease_confidence = float(disease_probs[disease_idx] * 100)
        disease_label = DISEASE_LABELS[disease_idx] if disease_idx < len(DISEASE_LABELS) else "Unknown"
        
        print(f"[OK] Disease: {disease_label} (confidence: {disease_confidence:.2f}%)")
        
        # Get all probabilities for display
        all_probabilities = {
            DISEASE_LABELS[i]: float(disease_probs[i] * 100)
            for i in range(min(len(DISEASE_LABELS), len(disease_probs)))
        }
        
        print(f"\n{'='*60}")
        print(f"FINAL DIAGNOSIS:")
        print(f"  Disease: {disease_label} ({disease_confidence:.2f}%)")
        print(f"  All probabilities: {all_probabilities}")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'error': None,
            'disease_label': disease_label,
            'disease_confidence': disease_confidence,
            'all_probabilities': all_probabilities
        }
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"\n✗ {error_msg}")
        traceback.print_exc()
        return {
            'success': False,
            'error': error_msg,
            'disease_label': 'Unknown',
            'disease_confidence': 0.0
        }
# ===================================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-for-academic-project'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lung_sound.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VISUALIZATIONS_FOLDER'] = 'static/visualizations' # New folder for visualizations
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@login_manager.user_loader
def load_user(user_id):
        return User.query.get(int(user_id))

def is_admin():
    return current_user.is_authenticated and current_user.role == 'admin'

def generate_waveform_plot(audio_path, output_path, sr=MEL_PARAMS['target_sr']):
    """Generates and saves a waveform plot."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        plt.figure(figsize=(10, 4))

        # Fix for matplotlib compatibility - create axes first
        ax = plt.gca()
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#4A90E2')

        plt.title('Audio Waveform', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=11)
        plt.ylabel('Amplitude', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[OK] Waveform plot saved to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Error generating waveform plot: {e}")
        traceback.print_exc()
        return False

def generate_spectrogram_plot(audio_path, output_path, sr=MEL_PARAMS['target_sr'], n_mels=MEL_PARAMS['n_mels'], n_fft=MEL_PARAMS['n_fft'], hop_length=MEL_PARAMS['hop_length']):
    """Generates and saves a mel spectrogram plot."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB', label='Intensity (dB)')
        plt.title('Mel Spectrogram (Frequency Analysis)', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=11)
        plt.ylabel('Frequency (Hz)', fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[OK] Spectrogram plot saved to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Error generating spectrogram plot: {e}")
        traceback.print_exc()
        return False

def check_patient_access(patient):
    if is_admin():
        return True
    if patient.clinician_id != current_user.id:
        abort(403)
    return True

def get_clinician_patients():
    if is_admin():
        return Patient.query.all()
    return Patient.query.filter_by(clinician_id=current_user.id).all()

def get_clinician_analyses():
    if is_admin():
        return AnalysisResult.query.order_by(AnalysisResult.analysis_date.desc()).all()
    return AnalysisResult.query.join(AudioFile).join(Patient).filter(
        Patient.clinician_id == current_user.id
    ).order_by(AnalysisResult.analysis_date.desc()).all()

def require_admin():
    if not is_admin():
        abort(403)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        phone_number = request.form.get('phone_number')
        password = request.form.get('password')

        # Security: Prevent registration with admin email domains or admin keywords
        if 'admin' in email.lower():
            flash('Invalid email address. Admin accounts cannot be created through registration.', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return redirect(url_for('register'))

        # Validate phone number is exactly 10 digits
        if not phone_number or not phone_number.isdigit() or len(phone_number) != 10:
            flash('Phone number must be exactly 10 digits.', 'error')
            return redirect(url_for('register'))

        # All public registrations are clinicians only
        new_user = User(
            username=username,
            email=email,
            phone_number=phone_number,
            password_hash=generate_password_hash(password),
            role='clinician'  # Explicitly set role to clinician
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            # Check if 2FA is enabled for this user
            if user.is_2fa_enabled:
                # Store user ID in session for 2FA verification
                session['pending_2fa_user_id'] = user.id
                return redirect(url_for('verify_2fa'))
            else:
                # No 2FA, login directly
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
        flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    # Clear 2FA session data if present
    session.pop('pending_2fa_user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ==================== Two-Factor Authentication Routes ====================

@app.route('/setup-2fa', methods=['GET', 'POST'])
@login_required
def setup_2fa():
    """Setup page for enabling 2FA"""
    if request.method == 'POST':
        verification_code = request.form.get('verification_code')

        # Verify the code
        totp = pyotp.TOTP(current_user.totp_secret)
        if totp.verify(verification_code):
            # Enable 2FA for the user
            current_user.is_2fa_enabled = True
            db.session.commit()
            flash('Two-Factor Authentication has been enabled successfully!', 'success')
            return redirect(url_for('settings'))
        else:
            flash('Invalid verification code. Please try again.', 'error')
            # Keep the same secret and QR code
            totp_secret = current_user.totp_secret
    else:
        # Generate new secret if user doesn't have one
        if not current_user.totp_secret:
            totp_secret = pyotp.random_base32()
            current_user.totp_secret = totp_secret
            db.session.commit()
        else:
            totp_secret = current_user.totp_secret

    # Generate QR code
    totp_uri = pyotp.totp.TOTP(totp_secret).provisioning_uri(
        name=current_user.email,
        issuer_name='AST Lung Sound Analysis'
    )

    # Create QR code image
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to base64 for embedding in HTML
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    qr_code_b64 = base64.b64encode(buffer.getvalue()).decode()

    return render_template('setup_2fa.html',
                         qr_code_b64=qr_code_b64,
                         totp_secret=totp_secret)

@app.route('/verify-2fa', methods=['GET', 'POST'])
def verify_2fa():
    """Verification page during login for users with 2FA enabled"""
    # Check if there's a pending 2FA user
    pending_user_id = session.get('pending_2fa_user_id')
    if not pending_user_id:
        flash('No pending authentication. Please login first.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        verification_code = request.form.get('verification_code')
        user = User.query.get(pending_user_id)

        if not user:
            session.pop('pending_2fa_user_id', None)
            flash('Invalid session. Please login again.', 'error')
            return redirect(url_for('login'))

        # Verify the TOTP code
        totp = pyotp.TOTP(user.totp_secret)
        if totp.verify(verification_code):
            # Code is valid, complete the login
            session.pop('pending_2fa_user_id', None)
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid verification code. Please try again.', 'error')

    return render_template('verify_2fa.html')

@app.route('/disable-2fa', methods=['POST'])
@login_required
def disable_2fa():
    """Disable 2FA for the current user"""
    current_user.is_2fa_enabled = False
    current_user.totp_secret = None
    db.session.commit()
    flash('Two-Factor Authentication has been disabled.', 'info')
    return redirect(url_for('settings'))

# ==================== End of 2FA Routes ====================

@app.route('/patients')
@login_required
def patients_list():
    patients = get_clinician_patients()
    return render_template('patients.html', patients=patients)

@app.route('/patient/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    check_patient_access(patient)
    analyses = AnalysisResult.query.join(AudioFile).filter(AudioFile.patient_id == patient_id).order_by(AnalysisResult.analysis_date.desc()).all()
    return render_template('patient_detail.html', patient=patient, analyses=analyses)

@app.route('/dashboard')
@login_required
def dashboard():
    # Admins are redirected to admin dashboard, they cannot perform diagnoses
    if is_admin():
        return redirect(url_for('admin_dashboard'))

    recent_analyses = get_clinician_analyses()[:10]
    return render_template('dashboard.html', recent_analyses=recent_analyses)

@app.route('/upload', methods=['POST'])
@login_required
def upload_audio():
    # Prevent admins from uploading/diagnosing - they are system overseers only
    if is_admin():
        return jsonify({
            'success': False,
            'error': 'Access denied',
            'details': 'Administrators cannot perform patient diagnoses. Only clinicians can upload and analyze audio files.'
        }), 403

    try:
        print("\n" + "="*60)
        print("NEW AUDIO UPLOAD REQUEST")
        print("="*60)
        
        # Step 1: Get form data
        try:
            # Patient Identification
            first_name = request.form.get('first_name')
            middle_name = request.form.get('middle_name', '')
            last_name = request.form.get('last_name')
            patient_id = request.form.get('patient_id')
            date_of_birth = request.form.get('date_of_birth')
            gender = request.form.get('gender')

            # Construct full patient name
            if middle_name:
                patient_name = f"{first_name} {middle_name} {last_name}"
            else:
                patient_name = f"{first_name} {last_name}"

            # Contact Information
            phone_number = request.form.get('phone_number')
            email_address = request.form.get('email_address')

            # Clinical Context
            recording_datetime = request.form.get('recording_datetime')
            visit_type = request.form.get('visit_type')

            # Recording Details
            recording_location = request.form.get('recording_location', 'chest')
            clinical_notes = request.form.get('clinical_notes')

            print(f"[OK] Form data received: Patient={patient_name}, ID={patient_id}")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Form data error: {str(e)}'}), 400

        # Step 2: Validate required fields
        if not all([first_name, last_name, patient_id]):
            return jsonify({'success': False, 'error': 'First name, last name, and Patient ID are required'}), 400

        # Validate patient ID is exactly 4 digits
        if not patient_id.isdigit() or len(patient_id) != 4:
            return jsonify({'success': False, 'error': 'Patient ID must be exactly 4 digits'}), 400

        # Validate phone number is exactly 10 digits
        if phone_number and (not phone_number.isdigit() or len(phone_number) != 10):
            return jsonify({'success': False, 'error': 'Phone number must be exactly 10 digits'}), 400
        
        # Step 3: Check file upload
        if 'audio_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['audio_file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file format. Please upload WAV, MP3, or FLAC'}), 400
        
        print(f"[OK] File received: {file.filename}")
        
        # Step 4: Get or create patient
        try:
            patient = Patient.query.filter_by(patient_id=patient_id, clinician_id=current_user.id).first()
            if not patient:
                # Parse date_of_birth if provided
                dob_parsed = None
                if date_of_birth:
                    try:
                        from datetime import datetime as dt
                        dob_parsed = dt.strptime(date_of_birth, '%Y-%m-%d').date()
                    except ValueError:
                        pass  # Invalid date format, skip

                patient = Patient(
                    patient_name=patient_name,
                    patient_id=patient_id,
                    date_of_birth=dob_parsed,
                    gender=gender,
                    phone_number=phone_number if phone_number else None,
                    email_address=email_address if email_address else None,
                    clinician_id=current_user.id
                )
                db.session.add(patient)
                db.session.commit()
                print(f"[OK] New patient created: {patient_name} (ID: {patient_id})")
            else:
                print(f"[OK] Existing patient found: {patient_name} (ID: {patient_id})")
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': f'Database error creating patient: {str(e)}'}), 500
        
        # Step 5: Save audio file
        try:
            filename = secure_filename(file.filename)
            unique_filename = f"{current_user.id}_{patient_id}_{datetime.now().timestamp()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(filepath)
            print(f"[OK] Audio file saved: {filepath}")
        except Exception as e:
            return jsonify({'success': False, 'error': f'File save error: {str(e)}'}), 500
        
        # Step 6: Create audio file record
        try:
            # Parse recording_datetime if provided
            recording_dt_parsed = None
            if recording_datetime:
                try:
                    from datetime import datetime as dt
                    recording_dt_parsed = dt.strptime(recording_datetime, '%Y-%m-%dT%H:%M')
                except ValueError:
                    recording_dt_parsed = datetime.now()  # Fallback to current time
            else:
                recording_dt_parsed = datetime.now()

            audio_file = AudioFile(
                filename=filename,
                file_path=filepath,
                recording_location=recording_location,
                recording_datetime=recording_dt_parsed,
                visit_type=visit_type if visit_type else None,
                clinical_notes=clinical_notes if clinical_notes else None,
                patient_id=patient.id
            )
            db.session.add(audio_file)
            db.session.commit()
            print(f"[OK] Audio file record created in database")
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': f'Database error saving audio record: {str(e)}'}), 500
        
        # Step 7: Generate and save visualizations
        waveform_filename = f"waveform_{os.path.splitext(unique_filename)[0]}.png"
        spectrogram_filename = f"spectrogram_{os.path.splitext(unique_filename)[0]}.png"

        waveform_path_full = os.path.join(app.config['VISUALIZATIONS_FOLDER'], waveform_filename)
        spectrogram_path_full = os.path.join(app.config['VISUALIZATIONS_FOLDER'], spectrogram_filename)

        # Ensure visualization directory exists
        os.makedirs(app.config['VISUALIZATIONS_FOLDER'], exist_ok=True)

        waveform_success = generate_waveform_plot(filepath, waveform_path_full)
        spectrogram_success = generate_spectrogram_plot(filepath, spectrogram_path_full)

        # Store relative paths for Flask's static file serving
        waveform_db_path = f"visualizations/{waveform_filename}" if waveform_success else None
        spectrogram_db_path = f"visualizations/{spectrogram_filename}" if spectrogram_success else None
        
        # Step 8: Make prediction
        try:
            print("\nStarting disease diagnosis prediction...")
            prediction_result = predict_disease_diagnosis(filepath)
            
            # Check if prediction was successful
            if not prediction_result.get('success', False):
                error_message = prediction_result.get('error', 'Unknown error during prediction')
                print(f"[ERROR] Prediction failed: {error_message}")
                
                # Still save analysis with error information
                try:
                    analysis = AnalysisResult(
                        classification='Error',
                        confidence_score=0.0,
                        disease_diagnosis='Error',
                        disease_confidence=0.0,
                        audio_file_id=audio_file.id,
                        waveform_path=waveform_db_path,
                        spectrogram_path=spectrogram_db_path
                    )
                    db.session.add(analysis)
                    db.session.commit()
                except:
                    db.session.rollback()
                
                return jsonify({
                    'success': False,
                    'error': error_message,
                    'details': 'The audio file could not be analyzed. Please ensure it is a valid respiratory sound recording.'
                }), 500
            
        except Exception as e:
            print(f"[ERROR] Exception during prediction: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Prediction error',
                'details': str(e)
            }), 500
        
        # Step 9: Save successful analysis
        try:
            disease_label = prediction_result.get('disease_label', 'Unknown')
            disease_confidence = prediction_result.get('disease_confidence', 0.0)
            
            print(f"\n[OK] Prediction successful:")
            print(f"  - Disease: {disease_label} ({disease_confidence:.2f}%)")

            analysis = AnalysisResult(
                classification=disease_label,
                confidence_score=disease_confidence,
                disease_diagnosis=disease_label,
                disease_confidence=disease_confidence,
                audio_file_id=audio_file.id,
                waveform_path=waveform_db_path,
                spectrogram_path=spectrogram_db_path
            )
            db.session.add(analysis)
            db.session.commit()

            print(f"[OK] Analysis saved to database (ID: {analysis.id})")
            print("="*60 + "\n")
            
            return jsonify({
                'success': True,
                'analysis_id': analysis.id,
                'disease': disease_label,
                'disease_confidence': round(disease_confidence, 2),
                'all_probabilities': prediction_result.get('all_probabilities', {})
            })
            
        except Exception as e:
            db.session.rollback()
            print(f"[ERROR] Error saving analysis: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Failed to save analysis results',
                'details': str(e)
            }), 500
    
    except Exception as e:
        error_msg = f"Unexpected error in upload_audio: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        traceback.print_exc()
        print("="*60 + "\n")
        
        # Try to rollback any database changes
        try:
            db.session.rollback()
        except:
            pass
        
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred during analysis',
            'details': str(e)
        }), 500

@app.route('/results/<int:analysis_id>')
@login_required
def view_results(analysis_id):
    analysis = AnalysisResult.query.get_or_404(analysis_id)
    patient = analysis.audio_file.patient
    check_patient_access(patient)

    # Get clinician information
    clinician = patient.clinician

    results_data = {
        'disease_diagnosis': analysis.disease_diagnosis,
        'disease_confidence': analysis.disease_confidence,
        'clinical_interpretation': f'{analysis.disease_diagnosis} Diagnosis',
        'clinical_detail': f'Respiratory analysis indicates {analysis.disease_diagnosis} with {analysis.disease_confidence:.1f}% confidence.',
        'patient_name': patient.patient_name,
        'patient_id': patient.patient_id,
        'age': patient.age,
        'gender': patient.gender,
        'filename': analysis.audio_file.filename,
        'recording_location': analysis.audio_file.recording_location,
        'analysis_date': analysis.analysis_date.strftime('%B %d, %Y at %I:%M %p'),
        'file_size': '3.2 MB', # This is a placeholder, actual file size not calculated
        'classification': analysis.classification,
        'confidence_score': analysis.confidence_score,
        'waveform_path': analysis.waveform_path,
        'spectrogram_path': analysis.spectrogram_path,
        'attention_map_path': None, # Not implemented yet
        # Clinician contact information
        'clinician_name': clinician.username,
        'clinician_email': clinician.email,
        'clinician_phone': clinician.phone_number if clinician.phone_number else 'Not provided'
    }

    return render_template('results.html', **results_data)

@app.route('/reports')
@login_required
def reports():
    # Admin sees all analyses, clinicians see only their own
    if is_admin():
        total_analyses = AnalysisResult.query.count()

        disease_counts = db.session.query(
            AnalysisResult.disease_diagnosis,
            func.count(AnalysisResult.disease_diagnosis)
        ).group_by(AnalysisResult.disease_diagnosis).all()

        analyses = AnalysisResult.query.order_by(AnalysisResult.analysis_date.desc()).all()
    else:
        total_analyses = AnalysisResult.query.join(AudioFile).join(Patient).filter(
            Patient.clinician_id == current_user.id
        ).count()

        disease_counts = db.session.query(
            AnalysisResult.disease_diagnosis,
            func.count(AnalysisResult.disease_diagnosis)
        ).join(AudioFile).join(Patient).filter(
            Patient.clinician_id == current_user.id
        ).group_by(AnalysisResult.disease_diagnosis).all()

        analyses = AnalysisResult.query.join(AudioFile).join(Patient).filter(
            Patient.clinician_id == current_user.id
        ).order_by(AnalysisResult.analysis_date.desc()).all()

    disease_stats = {disease: count for disease, count in disease_counts}

    stats = {
        'total_analyses': total_analyses,
        'normal_count': disease_stats.get('Healthy', 0),
        'asthma_count': disease_stats.get('Asthma', 0),
        'copd_count': disease_stats.get('COPD', 0),
        'pneumonia_count': disease_stats.get('Pneumonia', 0),
        'bronchitis_count': disease_stats.get('Bronchitis', 0),
        'bronchiectasis_count': disease_stats.get('Bronchiectasis', 0),
        'urti_count': disease_stats.get('URTI', 0)
    }

    return render_template('reports.html', stats=stats, analyses=analyses)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        # Get form data
        new_username = request.form.get('username')
        new_email = request.form.get('email')
        new_phone_number = request.form.get('phone_number')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # Track if any changes were made
        changes_made = False

        # Update username if provided and different
        if new_username and new_username != current_user.username:
            current_user.username = new_username
            changes_made = True

        # Update email if provided and different
        if new_email and new_email != current_user.email:
            # Check if email contains 'admin' keyword (security check)
            if 'admin' in new_email.lower() and current_user.role != 'admin':
                flash('Invalid email address. Admin email addresses are restricted.', 'error')
                return redirect(url_for('settings'))

            # Check if email is already taken by another user
            existing_user = User.query.filter_by(email=new_email).first()
            if existing_user and existing_user.id != current_user.id:
                flash('Email already in use by another account.', 'error')
                return redirect(url_for('settings'))

            current_user.email = new_email
            changes_made = True

        # Update phone number if provided and different
        if new_phone_number and new_phone_number != current_user.phone_number:
            # Validate phone number is exactly 10 digits
            if not new_phone_number.isdigit() or len(new_phone_number) != 10:
                flash('Phone number must be exactly 10 digits.', 'error')
                return redirect(url_for('settings'))

            current_user.phone_number = new_phone_number
            changes_made = True

        # Update password if provided
        if new_password:
            # Verify current password is provided
            if not current_password:
                flash('Current password is required to change password.', 'error')
                return redirect(url_for('settings'))

            # Verify current password is correct
            if not check_password_hash(current_user.password_hash, current_password):
                flash('Current password is incorrect.', 'error')
                return redirect(url_for('settings'))

            # Verify new passwords match
            if new_password != confirm_password:
                flash('New passwords do not match.', 'error')
                return redirect(url_for('settings'))

            # Verify password strength (minimum 6 characters)
            if len(new_password) < 6:
                flash('New password must be at least 6 characters long.', 'error')
                return redirect(url_for('settings'))

            # Update password
            current_user.password_hash = generate_password_hash(new_password)
            changes_made = True

        # Commit changes if any were made
        if changes_made:
            try:
                db.session.commit()
                flash('Settings updated successfully!', 'success')
            except Exception as e:
                db.session.rollback()
                flash(f'Error updating settings: {str(e)}', 'error')
        else:
            flash('No changes were made.', 'info')

        return redirect(url_for('settings'))

    return render_template('settings.html')

@app.route('/clinical-guidelines')
@login_required
def clinical_guidelines():
    """Clinical guidelines page with disease information"""
    return render_template('clinical_guidelines.html')

@app.route('/admin')
@login_required
def admin_dashboard():
    require_admin()
    total_users = User.query.filter_by(role='clinician').count()
    total_patients = Patient.query.count()
    total_analyses = AnalysisResult.query.count()
    recent_analyses = AnalysisResult.query.order_by(AnalysisResult.analysis_date.desc()).limit(10).all()
    return render_template('admin_dashboard.html', total_users=total_users, total_patients=total_patients, total_analyses=total_analyses, recent_analyses=recent_analyses)

@app.route('/admin/clinicians')
@login_required
def admin_clinicians():
    require_admin()
    clinicians = User.query.filter_by(role='clinician').all()

    # Get patient count for each clinician
    clinician_stats = []
    for clinician in clinicians:
        patient_count = Patient.query.filter_by(clinician_id=clinician.id).count()
        analysis_count = AnalysisResult.query.join(AudioFile).join(Patient).filter(
            Patient.clinician_id == clinician.id
        ).count()
        clinician_stats.append({
            'clinician': clinician,
            'patient_count': patient_count,
            'analysis_count': analysis_count
        })

    return render_template('admin_clinicians.html', clinician_stats=clinician_stats)

@app.route('/admin/patients')
@login_required
def admin_patients():
    require_admin()
    patients = Patient.query.all()
    clinicians = User.query.filter_by(role='clinician').all()
    return render_template('admin_patients.html', patients=patients, clinicians=clinicians)

@app.route('/admin/reassign-patient', methods=['POST'])
@login_required
def admin_reassign_patient():
    require_admin()
    patient_id = request.form.get('patient_id')
    new_clinician_id = request.form.get('clinician_id')
    patient = Patient.query.get_or_404(patient_id)
    old_clinician_name = patient.clinician.username
    patient.clinician_id = new_clinician_id
    db.session.commit()
    flash(f'Patient reassigned from Dr. {old_clinician_name} to new clinician', 'success')
    return redirect(url_for('admin_patients'))

@app.route('/create-admin')
def create_admin():
    # Security: Require a secret key to create admin
    # In production, this should be an environment variable
    secret_key = request.args.get('secret')
    ADMIN_CREATION_SECRET = 'LUNG_ADMIN_2024_SECRET_KEY'  # Change this in production!

    if secret_key != ADMIN_CREATION_SECRET:
        return "Unauthorized: Invalid secret key", 403

    # Check if admin already exists
    admin = User.query.filter_by(email='admin@lunganalysis.com').first()
    if admin:
        # Update existing user to admin role if they exist
        admin.role = 'admin'
        db.session.commit()
        return f"User updated to admin role! Email: admin@lunganalysis.com"

    # Create new admin
    admin = User(
        username='System Admin',
        email='admin@lunganalysis.com',
        password_hash=generate_password_hash('admin123'),
        role='admin'
    )
    db.session.add(admin)
    db.session.commit()
    return "Admin created! Email: admin@lunganalysis.com, Password: admin123 (CHANGE THIS IMMEDIATELY!)"

@app.errorhandler(403)
def forbidden(e):
    flash('Access denied.', 'error')
    return redirect(url_for('dashboard'))

@app.errorhandler(404)
def not_found(e):
    flash('Not found.', 'error')
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    with app.app_context():
        print("Dropping old tables...")
        db.drop_all()
        print("Creating new tables...")
        db.create_all()
        print("[OK] Database initialized with fresh schema")
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        print("[OK] Upload folder ready")
        
        visualizations_folder = app.config['VISUALIZATIONS_FOLDER']
        if not os.path.exists(visualizations_folder):
            os.makedirs(visualizations_folder)
        print("[OK] Visualizations folder ready")
    print("\n" + "=" * 50)
    print("Lung Sound Analysis System Starting...")
    print("=" * 50)
    print("Open your browser to: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True)
