from fastapi import APIRouter, Depends, HTTPException
from app.schemas.auth import LoginRequest, SignUpRequest
from app.services.auth_service import AuthService
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter()
security = HTTPBearer()

@router.post("/login")
async def login(credentials: LoginRequest):
    return await AuthService.login_user(
        email=credentials.email,
        password=credentials.password
    )

@router.post("/signup")
async def signup(user_data: SignUpRequest):
    return await AuthService.create_user(
        email=user_data.email,
        password=user_data.password,
        display_name=user_data.display_name
    )

@router.get("/validate")
async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Validates the provided authentication token.
    Returns the decoded token information if valid, raises 401 if invalid.
    """
    return await AuthService.verify_token(credentials)

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Logs out the user by revoking their refresh tokens.
    Requires a valid authentication token.
    """
    decoded_token = await AuthService.verify_token(credentials)
    return await AuthService.logout_user(decoded_token["uid"]) 