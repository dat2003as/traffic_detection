"""
Global Error Handler Middleware
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import traceback
from app.settings import settings
logger = logging.getLogger(__name__)

# app/middlewares/error_handler.py

"""
Global Error Handler Middleware
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handler middleware."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except Exception as exc:
            # Log the error
            logger.error(
                f"Unhandled exception: {exc}",
                exc_info=True,
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "client": request.client.host if request.client else None
                }
            )
            
            # Determine status code
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            
            # Determine error message
            error_detail = str(exc)
            
            # In production, hide internal error details
            # ← FIX: Access settings from app.state or import directly
            if not settings.DEBUG:
                error_detail = "An internal server error occurred"
            
            # Return error response
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": "Internal Server Error",
                    "detail": error_detail,
                    "path": str(request.url.path),
                }
            )