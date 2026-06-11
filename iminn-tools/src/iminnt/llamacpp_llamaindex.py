"""
LlamaIndex integration for LlamaCppService.

This module provides LlamaIndex-compatible LLM wrapper for RISC-V llama.cpp
inference via QEMU. It allows using extended RISC-V CPU with LlamaIndex
frameworks to verify real-world LLM application support, particularly for
RAG (Retrieval Augmented Generation) workflows.

Phase 5: LlamaIndex integration for RAG framework verification.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field

try:
    # Try llama_index.core (newer versions >= 0.10.0)
    from llama_index.core.llms import (
        CustomLLM,
        CompletionResponse,
        CompletionResponseGen,
        LLMMetadata
    )
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    try:
        # Try direct import (older versions < 0.10.0)
        from llama_index.llms import CustomLLM
        from llama_index.llms.types import CompletionResponse, CompletionResponseGen
        LLAMAINDEX_AVAILABLE = True
    except ImportError:
        try:
            # Try alternative import paths
            from llama_index.llms.custom import CustomLLM
            from llama_index.llms.types import CompletionResponse, CompletionResponseGen
            LLAMAINDEX_AVAILABLE = True
        except ImportError:
            LLAMAINDEX_AVAILABLE = False
            CustomLLM = None
            CompletionResponse = None
            CompletionResponseGen = None

if not LLAMAINDEX_AVAILABLE:
    raise ImportError(
        "LlamaIndex not installed. Install with: pip install llama-index"
    )

from .llamacpp_service import LlamaCppService
from .log_cfg import logger


class RISCVRISCLLM(CustomLLM):
    """
    LlamaIndex LLM wrapper for RISC-V llama.cpp via QEMU.
    
    This class wraps LlamaCppService to provide a LlamaIndex-compatible
    interface. It allows LlamaIndex applications to use RISC-V compiled
    llama.cpp inference, enabling verification that extended RISC-V CPUs
    can support real-world RAG application frameworks.
    
    Example:
        >>> from iminnt.llamacpp_llamaindex import RISCVRISCLLM
        >>> from llama_index import VectorStoreIndex, SimpleDirectoryReader
        >>> 
        >>> llm = RISCVRISCLLM()
        >>> documents = SimpleDirectoryReader("data").load_data()
        >>> index = VectorStoreIndex.from_documents(documents)
        >>> query_engine = index.as_query_engine(llm=llm)
        >>> response = query_engine.query("What is the main topic?")
    """
    
    # Configuration fields (Pydantic)
    llama_cli_path: Optional[str] = Field(
        default=None,
        description="Path to llama-cli binary (uses default if None)"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to GGUF model file (uses default if None)"
    )
    max_tokens: int = Field(
        default=128,
        description="Maximum number of tokens to generate",
        ge=1,
        le=4096
    )
    threads: int = Field(
        default=1,
        description="Number of threads to use",
        ge=1,
        le=4
    )
    temperature: float = Field(
        default=0.8,
        description="Sampling temperature (0.0-2.0)",
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
        description="Additional llama-cli arguments"
    )
    
    # Internal service instance (not a Pydantic field)
    _service: Optional[LlamaCppService] = None
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize RISCVRISCLLM with optional configuration."""
        super().__init__(**kwargs)
        
        # Initialize LlamaCppService
        service_kwargs = {}
        if self.llama_cli_path:
            from pathlib import Path
            service_kwargs["llama_cli_path"] = Path(self.llama_cli_path)
        if self.model_path:
            from pathlib import Path
            service_kwargs["model_path"] = Path(self.model_path)
        
        try:
            self._service = LlamaCppService(**service_kwargs)
            logger.info(f"RISCVRISCLLM initialized with service: {self._service}")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaCppService: {e}")
            raise
    
    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata as LLMMetadata object."""
        return LLMMetadata(
            model_name="riscv_llamacpp",
            context_window=4096,  # Default context window for llama.cpp
            num_output=self.max_tokens,
            is_chat_model=False,
            is_function_calling_model=False,
        )
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Generate completion for a prompt.
        
        This is the required method for LlamaIndex CustomLLM.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
        
        Returns:
            CompletionResponse containing generated text
        """
        if self._service is None:
            raise RuntimeError(
                "LlamaCppService not initialized. "
                "Check initialization errors."
            )
        
        # Merge kwargs with instance parameters
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        threads = kwargs.get("threads", self.threads)
        temperature = kwargs.get("temperature", self.temperature)
        seed = kwargs.get("seed", self.seed)
        
        # Merge extra_args
        extra_args = self.extra_args.copy() if self.extra_args else {}
        if "extra_args" in kwargs:
            extra_args.update(kwargs.pop("extra_args", {}))
        
        try:
            # Generate text using the service
            response = self._service.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                threads=threads,
                temperature=temperature,
                seed=seed,
                **extra_args
            )
            
            # Create CompletionResponse
            completion_response = CompletionResponse(
                text=response,
                raw={"prompt": prompt, "response": response}
            )
            
            logger.debug(f"Generated completion (len={len(response)})")
            return completion_response
            
        except Exception as e:
            logger.error(f"Error during completion: {e}", exc_info=True)
            raise
    
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        Stream completion for a prompt.
        
        Note: Streaming is not yet supported in RISC-V llama-cli.
        This method falls back to non-streaming completion.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generator of CompletionResponse chunks
        """
        logger.warning(
            "Streaming not yet supported in RISC-V llama-cli. "
            "Falling back to non-streaming completion."
        )
        
        # Fallback to non-streaming
        response = self.complete(prompt, **kwargs)
        
        # Yield as a single chunk
        yield response
    
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> CompletionResponse:
        """
        Chat completion with message history.
        
        This is a simplified chat interface that concatenates messages
        into a single prompt.
        
        Args:
            messages: List of message dictionaries with "role" and "content"
            **kwargs: Additional generation parameters
        
        Returns:
            CompletionResponse containing generated text
        """
        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts)
        
        return self.complete(prompt, **kwargs)
