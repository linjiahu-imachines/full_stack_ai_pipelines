"""
FastAPI server for LlamaCppService.

This module provides HTTP API endpoints for the LlamaCppService, allowing
remote access to llama.cpp inference via QEMU user mode.

Phase 2: FastAPI server with REST API endpoints.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from pathlib import Path
import time
from .llamacpp_service import LlamaCppService
from .log_cfg import logger

# Initialize FastAPI app
app = FastAPI(
    title="LlamaCpp API",
    description="HTTP API for llama.cpp inference via QEMU RISC-V emulation",
    version="1.0.0",
)

# Global service instance (initialized on startup)
service: Optional[LlamaCppService] = None


# Request/Response Models
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(
        ...,
        description="Input text prompt for generation",
        min_length=1,
        max_length=10000
    )
    max_tokens: int = Field(
        default=128,
        description="Maximum number of tokens to generate",
        ge=1,
        le=4096
    )
    threads: int = Field(
        default=1,
        description="Number of threads to use (currently limited to 1 for QEMU user mode)",
        ge=1,
        le=4
    )
    temperature: float = Field(
        default=0.8,
        description="Sampling temperature (0.0-2.0). Higher = more creative",
        ge=0.0,
        le=2.0
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility",
        ge=0
    )
    extra_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional llama-cli arguments (advanced usage)"
    )
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    
    response: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    max_tokens: int = Field(..., description="Maximum tokens requested")
    threads: int = Field(..., description="Number of threads used")
    temperature: float = Field(..., description="Temperature used")
    seed: Optional[int] = Field(None, description="Random seed used")
    generation_time_ms: Optional[float] = Field(None, description="Generation time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    service_initialized: bool = Field(..., description="Whether service is initialized")
    timestamp: float = Field(..., description="Current timestamp")


# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize the LlamaCppService on server startup."""
    global service
    try:
        logger.info("Initializing LlamaCppService...")
        service = LlamaCppService()
        logger.info("LlamaCppService initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LlamaCppService: {e}")
        # Don't raise - allow server to start but endpoints will return errors
        service = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    global service
    logger.info("Shutting down LlamaCppService...")
    service = None


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LlamaCpp API",
        "version": "1.0.0",
        "description": "HTTP API for llama.cpp inference via QEMU RISC-V emulation",
        "docs": "/docs",
        "health": "/health",
        "generate": "/generate"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns the status of the API server and service.
    """
    return HealthResponse(
        status="healthy" if service is not None else "degraded",
        service_initialized=service is not None,
        timestamp=time.time()
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt.
    
    This endpoint uses the LlamaCppService to generate text via QEMU user mode.
    
    Args:
        request: GenerateRequest with prompt and generation parameters
    
    Returns:
        GenerateResponse with generated text and metadata
    
    Raises:
        HTTPException: If service is not initialized or generation fails
    """
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LlamaCppService is not initialized. Check server logs."
        )
    
    try:
        start_time = time.time()
        
        # Prepare kwargs for service.generate()
        kwargs = {}
        if request.extra_args:
            kwargs.update(request.extra_args)
        
        # Call the service
        response_text = service.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            threads=request.threads,
            temperature=request.temperature,
            seed=request.seed,
            **kwargs
        )
        
        generation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return GenerateResponse(
            response=response_text,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            threads=request.threads,
            temperature=request.temperature,
            seed=request.seed,
            generation_time_ms=round(generation_time, 2)
        )
        
    except ValueError as e:
        # Validation errors from service
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        # Missing files (binary, model, etc.)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Required file not found: {str(e)}"
        )
    except Exception as e:
        # Other errors
        logger.error(f"Error during generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@app.get("/info", response_model=Dict[str, Any])
async def info():
    """
    Get information about the service configuration.
    
    Returns paths and configuration information (without sensitive data).
    """
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LlamaCppService is not initialized"
        )
    
    return {
        "llama_cli_path": str(service.llama_cli_path),
        "model_path": str(service.model_path),
        "qemu_mode": "user" if service.use_qemu_user else "system",
        "binary_type": "static" if service._is_static else "dynamic"
    }


# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom exception handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if logger.level <= 10 else "Check server logs for details"
        }
    )
