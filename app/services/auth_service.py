from firebase_admin import auth
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict
import requests
from app.core.config import get_settings

settings = get_settings()
security = HTTPBearer()

class AuthService:
    @staticmethod
    async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
        try:
            decoded_token = auth.verify_id_token(credentials.credentials)
            return decoded_token
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )

    @staticmethod
    async def login_user(email: str, password: str) -> Dict:
        try:
            # Use Firebase Auth REST API to sign in with email/password
            response = requests.post(
                f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={settings.FIREBASE_WEB_API_KEY}",
                json={
                    "email": email,
                    "password": password,
                    "returnSecureToken": True
                }
            )
            
            if response.status_code != 200:
                error_message = response.json().get("error", {}).get("message", "Login failed")
                raise HTTPException(
                    status_code=401,
                    detail=error_message
                )

            auth_data = response.json()
            
            # Get user profile information
            user = auth.get_user(auth_data["localId"])
            # print(f"User: {user.UserRecord}")
            
            return {
                "message": "Login successful",
                "user_id": auth_data["localId"],
                "email": auth_data["email"],
                "display_name": user.display_name,
                "access_token": auth_data["idToken"],
                "refresh_token": auth_data["refreshToken"],
                "expires_in": auth_data["expiresIn"]
            }
            
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )

    @staticmethod
    async def create_user(email: str, password: str, display_name: Optional[str] = None) -> Dict:
        try:
            user = auth.create_user(
                email=email,
                password=password,
                display_name=display_name
            )
            return {
                "message": "User created successfully",
                "user_id": user.uid,
                "email": user.email,
                "display_name": user.display_name
            }
        except auth.EmailAlreadyExistsError:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )

    @staticmethod
    async def logout_user(user_id: str) -> Dict:
        try:
            auth.revoke_refresh_tokens(user_id)
            return {
                "message": "Logged out successfully",
                "user_id": user_id
            }
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            ) 