"""FastAPI server for main service API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared.logger import setup_logger
from main_service.api import routes

logger = setup_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Blockchain-Enabled Federated Learning - Main Service API",
    description="REST API for model management, training control, and monitoring",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.router)


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Main service API server starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Main service API server shutting down")


if __name__ == "__main__":
    import uvicorn
    from shared.config import settings

    port = getattr(settings, "api_port", 8000)
    host = getattr(settings, "api_host", "0.0.0.0")

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

