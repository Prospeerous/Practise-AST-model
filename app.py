from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime

from database.models import db, User, Patient, AudioFile, AnalysisResult

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-for-academic-project'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lung_sound.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
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

# Helper Functions
def is_admin():
    return current_user.is_authenticated and current_user.role == 'admin'

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

# Routes
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
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email, password_hash=generate_password_hash(password))
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
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    recent_analyses = get_clinician_analyses()[:10]
    return render_template('dashboard.html', recent_analyses=recent_analyses)

@app.route('/upload', methods=['POST'])
@login_required
def upload_audio():
    try:
        patient_name = request.form.get('patient_name')
        patient_id = request.form.get('patient_id')
        age = request.form.get('age')
        gender = request.form.get('gender')
        recording_location = request.form.get('recording_location', 'chest')
        
        if not all([patient_name, patient_id]):
            return jsonify({'error': 'Patient name and ID are required'}), 400
        
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['audio_file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        patient = Patient.query.filter_by(patient_id=patient_id, clinician_id=current_user.id).first()
        if not patient:
            patient = Patient(
                patient_name=patient_name,
                patient_id=patient_id,
                age=int(age) if age else None,
                gender=gender,
                clinician_id=current_user.id
            )
            db.session.add(patient)
            db.session.commit()
        
        filename = secure_filename(file.filename)
        unique_filename = f"{current_user.id}_{patient_id}_{datetime.now().timestamp()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        audio_file = AudioFile(filename=filename, file_path=filepath, recording_location=recording_location, patient_id=patient.id)
        db.session.add(audio_file)
        db.session.commit()
        
        # TODO: Replace with AST model prediction
        analysis = AnalysisResult(classification='Normal', confidence_score=95.0, audio_file_id=audio_file.id)
        db.session.add(analysis)
        db.session.commit()
        
        return jsonify({'success': True, 'analysis_id': analysis.id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<int:analysis_id>')
@login_required
def view_results(analysis_id):
    analysis = AnalysisResult.query.get_or_404(analysis_id)
    patient = analysis.audio_file.patient
    check_patient_access(patient)
    
    results_data = {
        'classification': analysis.classification,
        'confidence_score': analysis.confidence_score,
        'patient_name': patient.patient_name,
        'patient_id': patient.patient_id,
        'age': patient.age,
        'gender': patient.gender,
        'filename': analysis.audio_file.filename,
        'recording_location': analysis.audio_file.recording_location,
        'analysis_date': analysis.analysis_date.strftime('%B %d, %Y at %I:%M %p'),
        'file_size': '3.2 MB',
        'duration': '15 seconds',
        'spectrogram_path': None,
        'waveform_path': None,
        'attention_map_path': None
    }
    return render_template('results.html', **results_data)

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

@app.route('/admin')
@login_required
def admin_dashboard():
    require_admin()
    total_users = User.query.filter_by(role='clinician').count()
    total_patients = Patient.query.count()
    total_analyses = AnalysisResult.query.count()
    recent_analyses = AnalysisResult.query.order_by(AnalysisResult.analysis_date.desc()).limit(10).all()
    return render_template('admin_dashboard.html', total_users=total_users, total_patients=total_patients, total_analyses=total_analyses, recent_analyses=recent_analyses)

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
    admin = User.query.filter_by(email='admin@lunganalysis.com').first()
    if not admin:
        admin = User(username='System Admin', email='admin@lunganalysis.com', password_hash=generate_password_hash('admin123'), role='admin')
        db.session.add(admin)
        db.session.commit()
        return "Admin created! Email: admin@lunganalysis.com, Password: admin123"
    return "Admin already exists!"

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
        # Force drop and recreate all tables (FRESH START)
        print("Dropping old tables...")
        db.drop_all()
        print("Creating new tables...")
        db.create_all()
        print("✓ Database initialized with fresh schema")
        
        # Create upload folder
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        print("✓ Upload folder ready")
    
    print("\n" + "="*50)
    print("🫁 Lung Sound Analysis System Starting...")
    print("="*50)
    print("Open your browser to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=True)
@app.route('/reports')
@login_required
def reports():
    """Reports page - shows analysis statistics"""
    from collections import Counter
    
    # Get all analyses for current user
    analyses = get_clinician_analyses()
    
    # Calculate statistics
    total_analyses = len(analyses)
    classifications = [a.classification for a in analyses]
    classification_counts = Counter(classifications)
    
    stats = {
        'total_analyses': total_analyses,
        'normal_count': classification_counts.get('Normal', 0),
        'crackle_count': classification_counts.get('Crackle', 0),
        'wheeze_count': classification_counts.get('Wheeze', 0),
        'both_count': classification_counts.get('Both', 0),
    }
    
    return render_template('reports.html', stats=stats, analyses=analyses)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """Settings page - user profile settings"""
    if request.method == 'POST':
        # Update user settings
        new_username = request.form.get('username')
        if new_username:
            current_user.username = new_username
            db.session.commit()
            flash('Settings updated successfully!', 'success')
            return redirect(url_for('settings'))
    
    return render_template('settings.html')

