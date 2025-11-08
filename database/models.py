from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """Clinician/Doctor account"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)  # Clinician contact number
    role = db.Column(db.String(20), default='clinician')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Two-Factor Authentication fields
    totp_secret = db.Column(db.String(32), nullable=True)  # TOTP secret key
    is_2fa_enabled = db.Column(db.Boolean, default=False)  # 2FA enabled flag

    patients = db.relationship('Patient', backref='clinician', lazy=True, cascade='all, delete-orphan')

class Patient(db.Model):
    """Patient record"""
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_id = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))

    # Contact Information
    phone_number = db.Column(db.String(20), nullable=True)
    email_address = db.Column(db.String(120), nullable=True)

    clinician_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    audio_files = db.relationship('AudioFile', backref='patient', lazy=True, cascade='all, delete-orphan')

class AudioFile(db.Model):
    """Uploaded audio recording"""
    __tablename__ = 'audio_files'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(300), nullable=False)
    recording_location = db.Column(db.String(50))

    # Clinical Context
    recording_datetime = db.Column(db.DateTime, default=datetime.utcnow)
    visit_type = db.Column(db.String(50), nullable=True)  # Initial Assessment, Follow-up, Screening
    clinical_notes = db.Column(db.Text, nullable=True)  # Provider observations

    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)

    analysis_result = db.relationship('AnalysisResult', backref='audio_file', uselist=False, cascade='all, delete-orphan')

class AnalysisResult(db.Model):
    """AI Analysis results"""
    __tablename__ = 'analysis_results'
    
    id = db.Column(db.Integer, primary_key=True)
    classification = db.Column(db.String(100), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    disease_diagnosis = db.Column(db.String(100), nullable=True)
    disease_confidence = db.Column(db.Float, nullable=True)
    waveform_path = db.Column(db.String(300), nullable=True) # Path to waveform visualization
    spectrogram_path = db.Column(db.String(300), nullable=True) # Path to spectrogram visualization
    
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)
    audio_file_id = db.Column(db.Integer, db.ForeignKey('audio_files.id'), nullable=False)
