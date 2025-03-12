from pydantic import BaseModel
from typing import Optional

class FinancialQuery(BaseModel):
    prompt: str