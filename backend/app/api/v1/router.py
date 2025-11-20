from fastapi import APIRouter

from app.api.v1.endpoint import detection, video, statistics, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(detection.router, prefix="/detection",tags=["Detection"])

api_router.include_router(video.router,prefix="/video",tags=["Video Processing"])

api_router.include_router(  statistics.router,prefix="/statistics",tags=["Statistics & Analytics"])

api_router.include_router(health.router,prefix="/health",tags=["Health & Monitoring"])
