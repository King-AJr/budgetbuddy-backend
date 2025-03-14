from fastapi import APIRouter, Depends
from fastapi.security import HTTPBearer
from app.schemas.finance import FinancialQuery
from app.services.finance_service import FinanceService
from app.routes.auth import validate_token

router = APIRouter()
security = HTTPBearer()

@router.post("/chat")
async def analyze_finances(
    query: FinancialQuery,
    token: str = Depends(security)
):
    user = await validate_token(token)

    finance_service = FinanceService(user_id=user["uid"])
    result = finance_service.process_query(query.prompt)
    return result