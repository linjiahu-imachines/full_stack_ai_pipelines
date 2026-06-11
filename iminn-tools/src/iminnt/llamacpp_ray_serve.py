"""
Ray Serve deployment for RISC-V llama.cpp inference.

This module provides Ray Serve deployment class for RISC-V llama.cpp inference
via QEMU. It wraps LlamaCppService (Phase 1) to enable production-like deployment
patterns: scaling, load balancing, and resource isolation.

Phase 4: Ray Serve integration for deployment and scaling verification.
"""

from typing import Dict, Any, Optional
import asyncio
from fastapi import Request

try:
    from ray import serve
except ImportError:
    raise ImportError(
        "Ray Serve not installed. Install with: pip install 'ray[serve]'"
    )

from .llamacpp_service import LlamaCppService
from .log_cfg import logger


@serve.deployment(
    num_replicas=1,  # Default: 1 replica (can be overridden)
    ray_actor_options={"num_cpus": 1}  # CPU per replica
)
class RISCVRISCLLMDeployment:
    """
    Ray Serve deployment for RISC-V llama.cpp inference.
    
    This class wraps LlamaCppService to provide a Ray Serve deployment.
    It enables horizontal scaling by running multiple replicas, each with
    its own QEMU process for RISC-V inference.
    
    Example:
        >>> from ray import serve
        >>> from iminnt.llamacpp_ray_serve import RISCVRISCLLMDeployment
        >>> 
        >>> # Deploy with 2 replicas
        >>> deployment = RISCVRISCLLMDeployment.options(num_replicas=2)
        >>> serve.run(deployment)
    """
    
    def __init__(self):
        """
        Initialize service on replica creation.
        
        Each replica creates its own LlamaCppService instance, which
        means each replica has its own QEMU process for inference.
        """
        logger.info("Initializing RISCVRISCLLMDeployment replica...")
        try:
            self.service = LlamaCppService()
            logger.info("RISCVRISCLLMDeployment replica ready")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaCppService: {e}")
            raise
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """
        Handle incoming HTTP request.
        
        Expected JSON:
        {
            "prompt": "Hello, how are you?",
            "max_tokens": 128,
            "threads": 1,
            "temperature": 0.8,
            "seed": 42
        }
        
        Returns:
            JSON response with generated text and metadata
        """
        try:
            data = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {
                "error": f"Invalid JSON: {str(e)}",
                "status_code": 400
            }
        
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 128)
        threads = data.get("threads", 1)
        temperature = data.get("temperature", 0.8)
        seed = data.get("seed", 42)
        
        if not prompt:
            logger.warning("Request missing 'prompt' field")
            return {
                "error": "Missing 'prompt' field",
                "status_code": 400
            }
        
        logger.info(
            f"Processing request: prompt_len={len(prompt)}, "
            f"max_tokens={max_tokens}, threads={threads}"
        )
        
        try:
            # Generate using service (synchronous, but wrapped in async)
            # Use run_in_executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,  # Use default thread pool executor
                self.service.generate,
                prompt,
                max_tokens,
                threads,
                temperature,
                seed
            )
            
            logger.info(f"Request completed successfully (response_len={len(response)})")
            
            return {
                "response": response,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "threads": threads,
                "temperature": temperature,
                "seed": seed
            }
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return {
                "error": str(e),
                "status_code": 400
            }
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            return {
                "error": f"Internal server error: {str(e)}",
                "status_code": 500
            }
