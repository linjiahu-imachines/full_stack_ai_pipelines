# Framework Investigation: LangChain and Ray Serve

**Date:** January 6, 2025  
**Purpose:** Investigate LangChain and Ray Serve integration with QEMU RISC-V setup to verify extended RISC-V CPU can support real-world LLM application frameworks.

---

## Project Context

### Goal
Verify that the **new CPU with extended RISC-V instructions** can support real-world LLM applications that rely on:
- **Inference frameworks**: llama.cpp (✅ already working)
- **Application frameworks**: LangChain, Ray Serve, LlamaIndex, etc.

### Current Setup
- **QEMU RISC-V emulation** with IMI CPU model (`imicpu-v1`)
- **Extended RISC-V instructions** (IMI extensions)
- **llama.cpp** compiled for RISC-V with IMI support
- **Option A architecture**: Python frameworks on host, QEMU executes RISC-V binaries

### What We Need to Verify
1. Can LangChain workflows run with RISC-V llama.cpp inference?
2. Can Ray Serve deploy and scale RISC-V model inference?
3. Can these frameworks handle the QEMU overhead and still be practical?
4. What are the integration patterns and challenges?

---

## LangChain Investigation

### What is LangChain?
**LangChain** is a framework for building LLM applications with:
- **Chains**: Composable sequences of LLM calls
- **Agents**: LLMs that can use tools and make decisions
- **Memory**: Conversation history management
- **Vector Stores**: RAG (Retrieval Augmented Generation)
- **Document Loaders**: Process various data sources

### Architecture Options for RISC-V Integration

#### Option 1: Custom LLM Wrapper (Recommended)
**Pattern:** LangChain on host → Custom LLM wrapper → QEMU RISC-V llama-cli

```python
from langchain.llms.base import LLM
from iminnt.llamacpp_service import LlamaCppService

class RISCVRISCLLM(LLM):
    """LangChain wrapper for RISC-V llama.cpp via QEMU."""
    
    def __init__(self):
        super().__init__()
        self.service = LlamaCppService()
    
    def _call(self, prompt: str, stop: list = None) -> str:
        """Generate text using RISC-V inference."""
        return self.service.generate(
            prompt=prompt,
            max_tokens=256,
            temperature=0.7
        )
    
    @property
    def _llm_type(self) -> str:
        return "riscv_llamacpp"
```

**Advantages:**
- ✅ LangChain runs on host (full Python ecosystem)
- ✅ Uses existing `LlamaCppService` (Phase 1)
- ✅ Simple integration
- ✅ Can use all LangChain features (chains, agents, memory)

**Challenges:**
- ⚠️ Each LLM call = QEMU process overhead (~3-4 seconds)
- ⚠️ No streaming support (yet)
- ⚠️ Slower than native inference

**Use Cases to Test:**
1. **Simple Chain**: Prompt → LLM → Output
2. **Agent with Tools**: LLM that can call functions
3. **RAG Pipeline**: Document loading → Vector store → LLM query
4. **Conversation Chain**: Multi-turn conversations with memory

### Integration Points

**1. Basic LLM Integration**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from iminnt.llamacpp_langchain import RISCVRISCLLM

llm = RISCVRISCLLM()
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short story about {topic}:"
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("a robot learning to paint")
```

**2. Agent with Tools**
```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for math calculations"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("What is 15 * 23?")
```

**3. RAG Pipeline**
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load and split documents
loader = TextLoader("document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

# Create vector store (runs on host)
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Query with RISC-V LLM
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
result = qa_chain.run("What is the document about?")
```

### Performance Considerations

**Bottlenecks:**
1. **QEMU Overhead**: ~3-4 seconds per inference call
2. **Model Loading**: Each call loads model from scratch (no caching yet)
3. **Chain Latency**: Multi-step chains multiply the overhead

**Optimization Strategies:**
1. **Caching**: Cache model state or responses
2. **Batch Processing**: Process multiple prompts together
3. **Async Execution**: Parallel chain steps where possible
4. **Streaming**: Return tokens as they're generated (future)

### Testing Plan

**Test 1: Basic Chain**
- Verify simple prompt → LLM → output works
- Measure latency
- Check output quality

**Test 2: Multi-Step Chain**
- Test chains with 2-3 LLM calls
- Verify state is maintained correctly
- Measure cumulative latency

**Test 3: Agent with Tools**
- Test agent making decisions
- Verify tool calling works
- Check reasoning quality

**Test 4: RAG Pipeline**
- Test document loading (host)
- Test vector search (host)
- Test LLM query (RISC-V via QEMU)
- Verify end-to-end works

---

## Ray Serve Investigation

### What is Ray Serve?
**Ray Serve** is a scalable model serving framework:
- **Deployment**: Deploy models as services
- **Scaling**: Auto-scale based on load
- **Batching**: Batch requests for efficiency
- **Multi-model**: Serve multiple models simultaneously
- **Load Balancing**: Distribute requests across replicas

### Architecture Options for RISC-V Integration

#### Option 1: Ray Serve on Host, QEMU Processes as Backend (Recommended)
**Pattern:** Ray Serve → Python backend → QEMU RISC-V llama-cli

```python
from ray import serve
from iminnt.llamacpp_service import LlamaCppService

@serve.deployment(num_replicas=2)
class RISCVRISCLLMDeployment:
    def __init__(self):
        self.service = LlamaCppService()
    
    async def __call__(self, request):
        prompt = request.json()["prompt"]
        response = self.service.generate(
            prompt=prompt,
            max_tokens=request.json().get("max_tokens", 128)
        )
        return {"response": response}

# Deploy
serve.run(RISCVRISCLLMDeployment.bind())
```

**Advantages:**
- ✅ Ray Serve runs on host (full ecosystem)
- ✅ Can scale horizontally (multiple QEMU processes)
- ✅ Load balancing and batching
- ✅ Production-ready deployment patterns

**Challenges:**
- ⚠️ Each replica = separate QEMU process (memory overhead)
- ⚠️ No shared model state (each replica loads model)
- ⚠️ QEMU overhead per request

#### Option 2: Ray Serve in Guest (Complex, Not Recommended)
**Pattern:** Ray Serve runs inside QEMU guest OS

**Challenges:**
- ❌ Limited Python packages in RISC-V guest
- ❌ Complex networking setup
- ❌ Harder to debug
- ❌ Performance overhead

### Integration Points

**1. Basic Deployment**
```python
from ray import serve
from fastapi import Request
from iminnt.llamacpp_service import LlamaCppService

@serve.deployment(
    route_prefix="/generate",
    num_replicas=2,
    ray_actor_options={"num_cpus": 1}
)
class LlamaCppDeployment:
    def __init__(self):
        self.service = LlamaCppService()
    
    async def __call__(self, request: Request):
        data = await request.json()
        response = self.service.generate(
            prompt=data["prompt"],
            max_tokens=data.get("max_tokens", 128)
        )
        return {"response": response}

# Deploy
serve.run(LlamaCppDeployment.bind())
```

**2. Batching for Efficiency**
```python
@serve.deployment(
    route_prefix="/generate",
    max_ongoing_requests=10
)
class BatchedLlamaCppDeployment:
    def __init__(self):
        self.service = LlamaCppService()
    
    @serve.batch(max_batch_size=4, batch_wait_timeout_s=1.0)
    async def batch_generate(self, prompts):
        # Process multiple prompts (if llama-cli supports batching)
        results = []
        for prompt in prompts:
            result = self.service.generate(prompt=prompt, max_tokens=128)
            results.append(result)
        return results
    
    async def __call__(self, request: Request):
        data = await request.json()
        result = await self.batch_generate([data["prompt"]])
        return {"response": result[0]}
```

**3. Multi-Model Deployment**
```python
@serve.deployment
class ModelRouter:
    def __init__(self):
        self.models = {
            "stories15M": LlamaCppService(
                model_path=Path("models/stories15M-q4_0.gguf")
            ),
            "lille130M": LlamaCppService(
                model_path=Path("models/lille-130m-instruct.gguf")
            )
        }
    
    async def __call__(self, request: Request):
        data = await request.json()
        model_name = data.get("model", "stories15M")
        prompt = data["prompt"]
        
        service = self.models[model_name]
        response = service.generate(prompt=prompt, max_tokens=128)
        return {"response": response, "model": model_name}
```

### Performance Considerations

**Scaling Strategy:**
- **Horizontal Scaling**: Add more replicas (each = QEMU process)
- **Resource Limits**: Each replica needs CPU/memory for QEMU
- **Load Balancing**: Ray Serve distributes requests
- **Batching**: Group requests to amortize QEMU overhead

**Resource Requirements:**
- Each replica: ~500MB-1GB RAM (QEMU + model)
- CPU: 1-2 cores per replica (QEMU emulation)
- Disk: Model files shared via 9p filesystem

### Testing Plan

**Test 1: Single Replica**
- Deploy one replica
- Test basic request handling
- Measure latency and throughput

**Test 2: Multiple Replicas**
- Deploy 2-4 replicas
- Test load balancing
- Measure aggregate throughput

**Test 3: Batching**
- Test request batching
- Measure efficiency gains
- Compare with single requests

**Test 4: Concurrent Requests**
- Send multiple concurrent requests
- Test Ray Serve queuing
- Measure response times

**Test 5: Auto-scaling**
- Test auto-scaling based on load
- Measure replica creation time
- Test scale-down behavior

---

## LlamaIndex Investigation

### What is LlamaIndex?
**LlamaIndex** is a framework for LLM data applications:
- **Data Connectors**: Load data from various sources
- **Indexing**: Create vector indices for RAG
- **Query Engines**: Query indexed data with LLMs
- **Agents**: LLM agents that can query data

### Integration Pattern

**Similar to LangChain:**
- LlamaIndex on host
- Custom LLM wrapper for RISC-V inference
- Vector operations on host
- LLM queries via QEMU RISC-V

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import CustomLLM
from iminnt.llamacpp_service import LlamaCppService

class RISCVRISCLLM(CustomLLM):
    def __init__(self):
        self.service = LlamaCppService()
    
    def complete(self, prompt: str, **kwargs) -> str:
        return self.service.generate(prompt=prompt, max_tokens=256)
    
    @property
    def metadata(self):
        return {"model_name": "riscv_llamacpp"}

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query with RISC-V LLM
query_engine = index.as_query_engine(llm=RISCVRISCLLM())
response = query_engine.query("What is this document about?")
```

---

## Verification Strategy

### What We're Verifying

1. **Functional Verification:**
   - ✅ Can LangChain chains execute with RISC-V inference?
   - ✅ Can Ray Serve deploy and scale RISC-V models?
   - ✅ Do these frameworks work end-to-end?

2. **Performance Verification:**
   - ⚠️ Is QEMU overhead acceptable for real applications?
   - ⚠️ Can we achieve reasonable latency?
   - ⚠️ Can we scale to handle multiple requests?

3. **Compatibility Verification:**
   - ✅ Do extended RISC-V instructions work correctly?
   - ✅ Are there any instruction-level issues?
   - ✅ Does the CPU model handle all operations?

### Success Criteria

**Minimum Viable:**
- LangChain basic chain works
- Ray Serve single replica works
- End-to-end latency < 10 seconds for simple queries

**Ideal:**
- LangChain agents work
- Ray Serve multi-replica with load balancing
- Latency < 5 seconds per request
- Can handle 10+ concurrent requests

**Stretch Goals:**
- Streaming support
- Model caching
- Batch processing
- Auto-scaling

---

## Recommended Starting Point: LangChain

### Why Start with LangChain?

**1. Most Representative of Real-World Applications**
- LangChain is the most widely used LLM application framework
- If your CPU can handle LangChain workflows, it demonstrates broad compatibility
- Covers the most common use cases (chains, agents, RAG)

**2. Progressive Complexity Testing**
- **Level 1**: Simple chain (prompt → LLM → output) - Quick verification
- **Level 2**: Multi-step chain - Test sequential operations
- **Level 3**: Agent with tools - Test decision-making and function calling
- **Level 4**: RAG pipeline - Test document processing + LLM queries
- This allows incremental verification from simple to complex

**3. Direct Integration Path**
- Can directly use existing `LlamaCppService` (Phase 1)
- Simple wrapper class needed (minimal code)
- No additional infrastructure required
- Easy to debug and test

**4. Best for CPU Verification**
- Tests actual application logic (not just infrastructure)
- Exercises CPU with real workloads
- Can verify extended RISC-V instructions work in context
- Demonstrates end-to-end application support

**5. Foundation for Others**
- If LangChain works, LlamaIndex will likely work (similar pattern)
- Ray Serve can be added later for deployment/scaling verification
- Establishes the integration pattern others can follow

### Comparison

| Framework | Complexity | Value for Verification | Integration Effort | Priority |
|-----------|------------|----------------------|-------------------|----------|
| **LangChain** | Medium | ⭐⭐⭐⭐⭐ Highest | Low | **Start Here** |
| Ray Serve | High | ⭐⭐⭐ Infrastructure | Medium | Phase 4 |
| LlamaIndex | Medium | ⭐⭐⭐ Similar to LangChain | Low | Phase 4+ |

### Phase 3 Plan: LangChain Integration

**Step 1: Create LangChain Wrapper** (1-2 hours)
- Implement `RISCVRISCLLM` class extending LangChain's `LLM` base
- Integrate with existing `LlamaCppService`
- Test basic `_call()` method

**Step 2: Test Simple Chain** (30 min)
- Create basic prompt → LLM → output chain
- Verify it works end-to-end
- Measure latency

**Step 3: Test Advanced Features** (1-2 hours)
- Multi-step chains
- Agent with tools
- RAG pipeline (if needed)

**Step 4: Document and Verify** (30 min)
- Document integration pattern
- Create examples
- Verify CPU handles all operations correctly

**Total Estimated Time: 3-5 hours**

### Why Not Ray Serve First?

- **More Complex**: Requires understanding Ray ecosystem, deployment patterns
- **Infrastructure Focus**: Tests deployment/scaling, not application logic
- **Less Direct**: More layers between your CPU and the actual LLM work
- **Can Come Later**: Once LangChain works, Ray Serve is a natural next step for production deployment

### Why Not LlamaIndex First?

- **Similar to LangChain**: Overlaps in functionality and integration pattern
- **Less Popular**: LangChain has more examples and community support
- **Can Add Later**: If LangChain works, LlamaIndex will likely work too

---

## Next Steps

**Phase 3: LangChain Integration** (Recommended Start)

1. **Create LangChain Wrapper** (`src/iminnt/llamacpp_langchain.py`)
   - Implement `RISCVRISCLLM` class
   - Test basic chain
   - Test agent with tools

2. **Create Test Scripts**
   - LangChain integration tests
   - Performance benchmarks
   - Use case examples

3. **Document Integration Patterns**
   - Update quickstart guide
   - Add examples
   - Document limitations

**Phase 4: Ray Serve** (After LangChain)

1. **Create Ray Serve Deployment** (`src/iminnt/llamacpp_ray_serve.py`)
   - Implement deployment class
   - Test single replica
   - Test multiple replicas

**Phase 5: LlamaIndex** (Optional, if needed)

1. **Create LlamaIndex Wrapper** (similar to LangChain)
2. **Test RAG workflows**

---

## Key Questions to Answer

1. **Can we use existing FastAPI server with LangChain?**
   - Yes, LangChain can call HTTP APIs
   - Or use direct service integration

2. **Should we implement streaming?**
   - Important for user experience
   - Requires async subprocess handling

3. **How to handle model caching?**
   - Keep QEMU processes alive?
   - Shared memory?
   - Redis cache?

4. **What about LlamaIndex?**
   - Similar pattern to LangChain
   - Can implement after LangChain

---

## References

- **LangChain Docs**: https://python.langchain.com/
- **Ray Serve Docs**: https://docs.ray.io/en/latest/serve/
- **LlamaIndex Docs**: https://docs.llamaindex.ai/
- **Current Setup**: `docs/option_a_quickstart.md`
- **Architecture Guide**: `docs/higher_level_frameworks_guide.md`
