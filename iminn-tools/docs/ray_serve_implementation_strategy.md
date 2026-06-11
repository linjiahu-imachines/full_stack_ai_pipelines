# Ray Serve Integration - Implementation Strategy

**Date:** January 6, 2025  
**Purpose:** Define clear implementation strategy for Phase 4 - Ray Serve integration with RISC-V llama.cpp via QEMU  
**Status:** Strategy Planning (Before Implementation)

---

## Executive Summary

**Goal:** Integrate Ray Serve to deploy and scale RISC-V llama.cpp inference, verifying that extended RISC-V CPU can support production-like deployment patterns (scaling, load balancing, batching).

**Approach:** Host-based Ray Serve deployment that wraps existing `LlamaCppService` (Phase 1) and can be used with LangChain (Phase 3).

**Key Verification:** Can Ray Serve handle multiple concurrent requests with QEMU RISC-V inference? Can we scale horizontally with multiple replicas?

---

## 1. Architecture Analysis

### 1.1 Current State

**What We Have:**
- ✅ **Phase 1**: `LlamaCppService` - Python wrapper for QEMU user mode llama-cli execution
- ✅ **Phase 2**: FastAPI server - HTTP API for inference
- ✅ **Phase 3**: LangChain integration - Application framework support
- ✅ **QEMU Setup**: User mode working, system mode available
- ✅ **Infrastructure**: Paths, constants, logging configured

**What We Need:**
- Ray Serve deployment layer
- Scaling and load balancing
- Request batching (optional)
- Integration with existing services

### 1.2 Architecture Options

#### **Option 1: Simple Deployment (Recommended for Phase 4)**

**Pattern:** Ray Serve → `LlamaCppService` (direct reuse)

```
┌─────────────────────────────────────────────────────┐
│                   Ray Serve                         │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │  @serve.deployment(num_replicas=2)            │ │
│  │  class RISCVRISCLLMDeployment:                 │ │
│  │      def __init__(self):                       │ │
│  │          self.service = LlamaCppService()      │ │
│  │      async def __call__(self, request):        │ │
│  │          return self.service.generate(...)     │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  Load Balancer                                      │
│  ├── Replica 1 → LlamaCppService → QEMU           │
│  └── Replica 2 → LlamaCppService → QEMU           │
└─────────────────────────────────────────────────────┘
```

**Pros:**
- ✅ Simple: Reuses existing `LlamaCppService`
- ✅ Minimal code changes
- ✅ Easy to test and debug
- ✅ Clear separation of concerns

**Cons:**
- ⚠️ Each replica = separate QEMU process
- ⚠️ No request batching (each request is independent)
- ⚠️ Memory overhead (multiple QEMU instances)

**Use Case:** Initial verification, testing scaling behavior

#### **Option 2: Batched Deployment**

**Pattern:** Ray Serve → Batched `LlamaCppService` wrapper

```
┌─────────────────────────────────────────────────────┐
│                   Ray Serve                         │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │  @serve.deployment                             │ │
│  │  @serve.batch(max_batch_size=4)                │ │
│  │  class BatchedRISCVRISCLLMDeployment:          │ │
│  │      async def batch_generate(self, prompts):  │ │
│  │          # Process multiple prompts            │ │
│  │          return [self.service.generate(p)...]  │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**Pros:**
- ✅ Better resource utilization
- ✅ Can amortize QEMU startup overhead
- ✅ Higher throughput under load

**Cons:**
- ⚠️ More complex (batching logic)
- ⚠️ llama-cli doesn't natively support batching
- ⚠️ Would need sequential processing anyway

**Use Case:** High-throughput scenarios (future optimization)

#### **Option 3: Multi-Model Deployment**

**Pattern:** Ray Serve → Router → Multiple model deployments

```
┌─────────────────────────────────────────────────────┐
│                   Ray Serve                         │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │  ModelRouter                                   │ │
│  │  ├── stories15M → Deployment 1                │ │
│  │  └── lille130M → Deployment 2                 │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**Pros:**
- ✅ Support multiple models
- ✅ Independent scaling per model
- ✅ Model-specific optimization

**Cons:**
- ⚠️ More complex setup
- ⚠️ More resource requirements
- ⚠️ May not be needed for verification

**Use Case:** Production with multiple models (future)

### 1.3 Recommended Approach

**Phase 4 Implementation: Option 1 (Simple Deployment)**

**Rationale:**
1. **Simplicity First**: Focus on verifying Ray Serve works with RISC-V inference
2. **Reuse Existing Code**: Leverage `LlamaCppService` from Phase 1
3. **Clear Verification**: Test scaling behavior (1 → 2 → 4 replicas)
4. **Foundation for Future**: Can add batching/multi-model later

**What We'll Verify:**
- ✅ Ray Serve can deploy RISC-V inference service
- ✅ Multiple replicas can handle concurrent requests
- ✅ Load balancing distributes requests correctly
- ✅ Resource isolation works (each replica = separate QEMU)
- ✅ Performance characteristics (latency, throughput)

---

## 2. Implementation Components

### 2.1 Core Components

#### **Component 1: Ray Serve Deployment Class**

**File:** `src/iminnt/llamacpp_ray_serve.py`

**Responsibilities:**
- Wrap `LlamaCppService` in Ray Serve deployment
- Handle HTTP requests (FastAPI-style or raw)
- Manage service lifecycle (initialization, cleanup)
- Integrate with Ray Serve configuration (replicas, resources)

**Key Methods:**
- `__init__()`: Initialize `LlamaCppService`
- `__call__()`: Handle incoming requests
- `reconfigure()`: Update deployment configuration (optional)

#### **Component 2: Deployment Script**

**File:** `scripts/deploy_ray_serve.py` or `scripts/run_ray_serve.sh`

**Responsibilities:**
- Start Ray cluster (if needed)
- Deploy the service
- Handle configuration (number of replicas, resources)
- Provide deployment status

#### **Component 3: Test Script**

**File:** `scripts/test_ray_serve.py`

**Responsibilities:**
- Test basic request handling
- Test scaling (1, 2, 4 replicas)
- Test concurrent requests
- Measure performance (latency, throughput)
- Verify load balancing

### 2.2 Integration Points

#### **With Phase 1 (`LlamaCppService`)**
- Direct reuse: `self.service = LlamaCppService()`
- No modifications needed
- All Phase 1 features available (parameters, error handling)

#### **With Phase 3 (LangChain)**
- LangChain can call Ray Serve HTTP endpoint
- Or: Use Ray Serve deployment directly (future)
- Maintains compatibility with existing LangChain integration

#### **With Phase 2 (FastAPI)**
- **Decision Point**: Replace FastAPI with Ray Serve? Or run both?
- **Recommendation**: Keep FastAPI for simple use, Ray Serve for scaling
- Both can coexist (different ports)

### 2.3 Dependencies

**Required:**
- `ray[serve]` >= 2.8.0 (Ray Serve framework)
- `requests` (for testing HTTP endpoints)
- Existing: `LlamaCppService` (Phase 1)

**Optional:**
- `ray[default]` (full Ray stack, if needed)
- `prometheus-client` (for metrics, future)

---

## 3. Implementation Plan

### 3.1 Phase 4.1: Basic Deployment (Foundation)

**Goal:** Deploy single replica, verify basic functionality

**Steps:**
1. Install Ray Serve
2. Create deployment class (`RISCVRISCLLMDeployment`)
3. Create deployment script
4. Test single replica
5. Verify HTTP endpoint works

**Files to Create:**
- `src/iminnt/llamacpp_ray_serve.py` (deployment class)
- `scripts/deploy_ray_serve.py` (deployment script)
- `scripts/test_ray_serve.py` (test script)

**Success Criteria:**
- ✅ Deployment starts successfully
- ✅ HTTP endpoint responds to requests
- ✅ Single request completes correctly
- ✅ Response time acceptable (~6-10 seconds)

**Estimated Time:** 1-2 hours

### 3.2 Phase 4.2: Scaling Verification (Core Goal)

**Goal:** Test multiple replicas, verify load balancing

**Steps:**
1. Configure deployment with 2 replicas
2. Send concurrent requests
3. Verify requests distributed across replicas
4. Measure performance (latency, throughput)
5. Test with 4 replicas (if resources allow)

**Files to Modify:**
- `scripts/deploy_ray_serve.py` (add replica configuration)
- `scripts/test_ray_serve.py` (add concurrent request testing)

**Success Criteria:**
- ✅ Multiple replicas start successfully
- ✅ Load balancing distributes requests
- ✅ Concurrent requests handled correctly
- ✅ Aggregate throughput increases with replicas
- ✅ Resource isolation confirmed (separate QEMU processes)

**Estimated Time:** 1-2 hours

### 3.3 Phase 4.3: Integration Testing (Optional)

**Goal:** Test integration with LangChain and real-world patterns

**Steps:**
1. Test LangChain → Ray Serve endpoint
2. Test multi-step workflows
3. Measure end-to-end latency
4. Test error handling and recovery

**Files to Modify:**
- `scripts/test_ray_serve.py` (add integration tests)

**Success Criteria:**
- ✅ LangChain can call Ray Serve endpoint
- ✅ Complex workflows execute correctly
- ✅ Error handling works as expected

**Estimated Time:** 1 hour

### 3.4 Phase 4.4: Documentation and Refinement

**Goal:** Document implementation, update guides

**Steps:**
1. Update `option_a_quickstart.md` with Phase 4 details
2. Document deployment commands
3. Document testing procedures
4. Note limitations and future work

**Files to Modify:**
- `docs/option_a_quickstart.md` (add Phase 4 section)

**Estimated Time:** 30 minutes

**Total Estimated Time: 3.5 - 5.5 hours**

---

## 4. Detailed Implementation Strategy

### 4.1 Deployment Class Design

**Pattern: Ray Serve Deployment Decorator**

```python
from ray import serve
from fastapi import Request
from typing import Dict, Any
from iminnt.llamacpp_service import LlamaCppService
from iminnt.log_cfg import logger

@serve.deployment(
    route_prefix="/generate",
    num_replicas=1,  # Start with 1, increase for scaling
    ray_actor_options={"num_cpus": 1}  # CPU per replica
)
class RISCVRISCLLMDeployment:
    """Ray Serve deployment for RISC-V llama.cpp inference."""
    
    def __init__(self):
        """Initialize service on replica creation."""
        logger.info("Initializing RISCVRISCLLMDeployment replica...")
        self.service = LlamaCppService()
        logger.info("RISCVRISCLLMDeployment replica ready")
    
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
        """
        data = await request.json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 128)
        threads = data.get("threads", 1)
        temperature = data.get("temperature", 0.8)
        seed = data.get("seed", 42)
        
        if not prompt:
            return {
                "error": "Missing 'prompt' field",
                "status_code": 400
            }
        
        try:
            # Generate using service (synchronous, but wrapped in async)
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.service.generate,
                prompt,
                max_tokens,
                threads,
                temperature,
                seed
            )
            
            return {
                "response": response,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "threads": threads
            }
            
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            return {
                "error": str(e),
                "status_code": 500
            }
```

**Key Design Decisions:**
1. **FastAPI Request**: Use `fastapi.Request` for compatibility (Ray Serve supports it)
2. **Async Wrapper**: Use `run_in_executor` to wrap synchronous `service.generate()`
3. **Error Handling**: Return JSON errors, maintain HTTP semantics
4. **Logging**: Use existing logger for consistency

### 4.2 Deployment Script Design

**Pattern: Python Script with Ray Serve API**

```python
#!/usr/bin/env python3
"""
Deploy RISC-V llama.cpp inference service via Ray Serve.
"""

import ray
from ray import serve
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iminnt.llamacpp_ray_serve import RISCVRISCLLMDeployment

def main():
    parser = argparse.ArgumentParser(
        description="Deploy RISC-V llama.cpp service via Ray Serve"
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="Number of replicas (default: 1)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Start Ray (if not already running)
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        print("✅ Connected to existing Ray cluster")
    except:
        ray.init()
        print("✅ Started new Ray cluster")
    
    # Deploy service
    print(f"Deploying RISCVRISCLLMDeployment with {args.replicas} replica(s)...")
    
    deployment = RISCVRISCLLMDeployment.options(
        num_replicas=args.replicas,
        route_prefix="/generate"
    )
    
    serve.run(deployment, host=args.host, port=args.port)
    
    print(f"✅ Service deployed at http://{args.host}:{args.port}/generate")
    print("Press Ctrl+C to stop")

if __name__ == "__main__":
    main()
```

**Key Design Decisions:**
1. **Ray Initialization**: Auto-connect or start new cluster
2. **Replica Configuration**: Command-line argument for flexibility
3. **Route Prefix**: `/generate` for consistency with Phase 2
4. **Host/Port**: Configurable for different environments

### 4.3 Test Script Design

**Pattern: Comprehensive Test Suite**

```python
#!/usr/bin/env python3
"""
Test Ray Serve deployment for RISC-V llama.cpp inference.
"""

import requests
import time
import concurrent.futures
import argparse
from typing import List, Dict, Any

def test_single_request(url: str) -> Dict[str, Any]:
    """Test single request."""
    print("Test 1: Single Request")
    start = time.time()
    
    response = requests.post(
        url,
        json={
            "prompt": "Hello, how are you?",
            "max_tokens": 32,
            "threads": 1
        },
        timeout=30
    )
    
    elapsed = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ Success ({elapsed:.2f}s)")
        print(f"  Response: {data.get('response', '')[:80]}...")
        return {"success": True, "elapsed": elapsed}
    else:
        print(f"  ❌ Failed: {response.status_code}")
        return {"success": False, "elapsed": elapsed}

def test_concurrent_requests(url: str, num_requests: int = 4) -> Dict[str, Any]:
    """Test concurrent requests."""
    print(f"\nTest 2: Concurrent Requests ({num_requests} requests)")
    
    def send_request(i: int):
        start = time.time()
        response = requests.post(
            url,
            json={
                "prompt": f"Request {i}: Tell me a short story",
                "max_tokens": 32,
                "threads": 1
            },
            timeout=60
        )
        elapsed = time.time() - start
        return {
            "request_id": i,
            "status": response.status_code,
            "elapsed": elapsed,
            "success": response.status_code == 200
        }
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_elapsed = time.time() - start
    
    successful = sum(1 for r in results if r["success"])
    avg_latency = sum(r["elapsed"] for r in results) / len(results)
    
    print(f"  ✅ {successful}/{num_requests} requests succeeded")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"  Average latency: {avg_latency:.2f}s")
    print(f"  Throughput: {num_requests/total_elapsed:.2f} req/s")
    
    return {
        "success": successful == num_requests,
        "total_elapsed": total_elapsed,
        "avg_latency": avg_latency,
        "throughput": num_requests/total_elapsed
    }

def main():
    parser = argparse.ArgumentParser(
        description="Test Ray Serve deployment"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/generate",
        help="Service URL (default: http://localhost:8000/generate)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="Number of concurrent requests (default: 4)"
    )
    
    args = parser.parse_args()
    
    print("Ray Serve Integration Test Suite")
    print(f"Testing endpoint: {args.url}\n")
    
    # Test 1: Single request
    result1 = test_single_request(args.url)
    
    # Test 2: Concurrent requests
    result2 = test_concurrent_requests(args.url, args.concurrent)
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Single Request: {'✅ PASSED' if result1['success'] else '❌ FAILED'}")
    print(f"Concurrent Requests: {'✅ PASSED' if result2['success'] else '❌ FAILED'}")
    
    if result1['success'] and result2['success']:
        print("\n🎉 All tests passed! Ray Serve integration working correctly!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

**Key Design Decisions:**
1. **Single Request Test**: Verify basic functionality
2. **Concurrent Request Test**: Verify scaling and load balancing
3. **Performance Metrics**: Latency, throughput measurements
4. **Flexible Configuration**: URL and concurrency configurable

---

## 5. Testing Strategy

### 5.1 Test Scenarios

#### **Scenario 1: Single Replica, Single Request**
- **Goal**: Verify basic deployment works
- **Steps**: Deploy 1 replica, send 1 request
- **Expected**: Request completes successfully (~6-10 seconds)

#### **Scenario 2: Single Replica, Concurrent Requests**
- **Goal**: Verify request queuing
- **Steps**: Deploy 1 replica, send 4 concurrent requests
- **Expected**: Requests processed sequentially, total time ~24-40 seconds

#### **Scenario 3: Multiple Replicas, Concurrent Requests**
- **Goal**: Verify load balancing and scaling
- **Steps**: Deploy 2 replicas, send 4 concurrent requests
- **Expected**: Requests distributed across replicas, total time ~12-20 seconds (2x speedup)

#### **Scenario 4: Multiple Replicas, High Concurrency**
- **Goal**: Verify resource isolation
- **Steps**: Deploy 4 replicas, send 8 concurrent requests
- **Expected**: All requests succeed, aggregate throughput increases

### 5.2 Success Criteria

**Functional:**
- ✅ All requests return correct responses
- ✅ Load balancing distributes requests across replicas
- ✅ Error handling works correctly
- ✅ Service can be stopped/restarted cleanly

**Performance:**
- ✅ Latency per request: ~6-10 seconds (QEMU overhead)
- ✅ Throughput scales with number of replicas
- ✅ No resource leaks (QEMU processes cleaned up)

**Verification:**
- ✅ Extended RISC-V CPU supports Ray Serve deployment patterns
- ✅ Horizontal scaling works with QEMU RISC-V inference
- ✅ Production-like patterns (load balancing, resource isolation) verified

---

## 6. Trade-offs and Considerations

### 6.1 Resource Requirements

**Per Replica:**
- CPU: 1-2 cores (for QEMU emulation)
- Memory: ~500MB-1GB (QEMU + model)
- Disk: Shared model files (via existing paths)

**Total (2 replicas):**
- CPU: 2-4 cores
- Memory: ~1-2GB
- **Note**: Host machine must have sufficient resources

### 6.2 Performance Characteristics

**Latency:**
- Single request: ~6-10 seconds (QEMU overhead)
- Per-replica latency: Same as single-request latency
- **Scaling Benefit**: Parallelism, not reduced latency

**Throughput:**
- Single replica: ~0.1-0.15 requests/second (sequential)
- N replicas: ~N * 0.1-0.15 requests/second (parallel)
- **Bottleneck**: QEMU emulation overhead

### 6.3 Limitations

**Known Limitations:**
1. **No Model Caching**: Each replica loads model separately
2. **QEMU Overhead**: Cannot be eliminated (by design)
3. **No Native Batching**: llama-cli processes one request at a time
4. **Resource Intensive**: Each replica = separate QEMU process

**Acceptable for Verification:**
- These limitations are acceptable for CPU verification purposes
- They demonstrate real-world constraints and trade-offs
- Can be optimized later (if needed)

### 6.4 Future Enhancements (Out of Scope)

**Potential Optimizations:**
1. **Request Batching**: Group multiple prompts (requires llama-cli batching support)
2. **Model Caching**: Shared model state (requires architecture change)
3. **Streaming Support**: Stream responses as generated (requires async subprocess)
4. **Metrics/Monitoring**: Prometheus/Grafana integration (requires additional setup)

**Recommendation:** Focus on core verification first, optimize later if needed.

---

## 7. Decision Points

### 7.1 Ray Serve vs FastAPI (Phase 2)

**Question:** Should we replace FastAPI with Ray Serve, or run both?

**Recommendation:** **Run Both**
- **FastAPI**: Simple, single-instance use cases
- **Ray Serve**: Scaling, production deployment patterns
- **Different Ports**: FastAPI on 8000, Ray Serve on 8001 (configurable)
- **Flexibility**: Choose based on use case

### 7.2 Synchronous vs Asynchronous Service Calls

**Question:** Should `LlamaCppService.generate()` be async?

**Recommendation:** **Keep Synchronous, Wrap in Executor**
- **Reason**: Minimal changes to Phase 1 code
- **Implementation**: Use `asyncio.run_in_executor()` in Ray Serve deployment
- **Future**: Can make service async later if needed

### 7.3 Replica Configuration

**Question:** How many replicas for testing?

**Recommendation:** **Start with 2, Test up to 4**
- **2 Replicas**: Clear scaling verification
- **4 Replicas**: Maximum reasonable for testing (resource limits)
- **Configurable**: Command-line argument for flexibility

---

## 8. Implementation Checklist

### Phase 4.1: Basic Deployment
- [ ] Install Ray Serve (`pip install "ray[serve]"`)
- [ ] Create `src/iminnt/llamacpp_ray_serve.py`
- [ ] Create `scripts/deploy_ray_serve.py`
- [ ] Create `scripts/test_ray_serve.py`
- [ ] Test single replica deployment
- [ ] Verify HTTP endpoint works

### Phase 4.2: Scaling Verification
- [ ] Test with 2 replicas
- [ ] Test concurrent requests (4 requests)
- [ ] Verify load balancing
- [ ] Measure performance (latency, throughput)
- [ ] Test with 4 replicas (optional)

### Phase 4.3: Integration Testing
- [ ] Test LangChain → Ray Serve endpoint
- [ ] Test error handling
- [ ] Verify resource cleanup

### Phase 4.4: Documentation
- [ ] Update `option_a_quickstart.md` with Phase 4 details
- [ ] Document deployment commands
- [ ] Document testing procedures
- [ ] Note limitations and future work

---

## 9. Next Steps

**Immediate Actions:**
1. Review and approve this strategy document
2. Proceed with Phase 4.1 (Basic Deployment)
3. Iterate based on results

**After Phase 4:**
- Consider batching optimization (if throughput needed)
- Consider multi-model deployment (if multiple models needed)
- Consider monitoring/metrics integration (if production use)

---

## 10. References

- **Ray Serve Docs**: https://docs.ray.io/en/latest/serve/
- **Ray Serve LLM Tutorial**: https://docs.ray.io/en/latest/serve/tutorials/deployment-serve-llm/
- **Existing Documentation**:
  - `docs/option_a_quickstart.md` (Phases 1-3)
  - `docs/framework_investigation.md` (Ray Serve section)
  - `docs/higher_level_frameworks_guide.md` (Ray Serve integration)

---

**End of Strategy Document**
