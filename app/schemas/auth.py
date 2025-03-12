from typing import Optional
from pydantic import BaseModel


class SignUpRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str