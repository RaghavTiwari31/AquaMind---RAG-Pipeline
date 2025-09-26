# =============================
# backend/app/routes/health.py
# =============================
from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "healthy", "message": "Backend is running"}