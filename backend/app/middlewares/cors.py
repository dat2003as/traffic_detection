"""
CORS Middleware Configuration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.settings import settings


def setup_cors(app: FastAPI) -> None:
    """
    Setup CORS middleware for the application.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
