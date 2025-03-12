import firebase_admin
from firebase_admin import credentials, firestore
from app.core.config import get_settings

settings = get_settings()

cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred)

db = firestore.client() 