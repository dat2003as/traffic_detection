"""
Main FastAPI application entry point.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
import logging

from app.settings import settings
from app.api.v1.router import api_router
from app.core.model import ModelManager
from app.db.database import close_db, init_db
from app.middlewares.cors import setup_cors
from app.middlewares.error_handler import ErrorHandlerMiddleware
from app.middlewares.rate_limit import RateLimitMiddleware

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    # Startup
    logger.info("Starting up...")
    
    # Load model
    model_manager = ModelManager()
    await model_manager.load_model()
    app.state.model_manager = model_manager
    
    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database init failed: {e}")
    
    yield
    
    # Shutdown
    await close_db()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered traffic violation detection system using YOLOv8",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


# CORS Middleware
setup_cors(app)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    requests_limit=10,       
    window_seconds=60      
)

app.mount("/storage", StaticFiles(directory="storage"), name="storage")

app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/api/docs",
        "health": "/api/v1/health"
    }


# Health check endpoint (also at root level)
@app.get("/health", tags=["Root"])
async def health_check():
    """Quick health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )