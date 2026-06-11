#!/usr/bin/env python3
"""
Deploy RISC-V llama.cpp inference service via Ray Serve.

This script starts Ray (if needed) and deploys the RISCVRISCLLMDeployment
service with configurable number of replicas.

Usage:
    python scripts/deploy_ray_serve.py
    python scripts/deploy_ray_serve.py --replicas 2
    python scripts/deploy_ray_serve.py --replicas 2 --port 8001
"""

import ray
from ray import serve
import argparse
from pathlib import Path
import sys
import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iminnt.llamacpp_ray_serve import RISCVRISCLLMDeployment
from iminnt.log_cfg import logger


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🛑 Shutting down Ray Serve...")
    try:
        serve.shutdown()
        ray.shutdown()
    except:
        pass
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy RISC-V llama.cpp service via Ray Serve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy with 1 replica (default)
  python scripts/deploy_ray_serve.py

  # Deploy with 2 replicas
  python scripts/deploy_ray_serve.py --replicas 2

  # Deploy with 2 replicas on port 8001
  python scripts/deploy_ray_serve.py --replicas 2 --port 8001
        """
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
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (default: start new cluster)"
    )
    
    args = parser.parse_args()
    
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start Ray (if not already running)
    try:
        if args.ray_address:
            ray.init(address=args.ray_address, ignore_reinit_error=True)
            print(f"✅ Connected to Ray cluster at {args.ray_address}")
        else:
            ray.init(ignore_reinit_error=True)
            print("✅ Connected to existing Ray cluster or started new cluster")
    except Exception as e:
        print(f"❌ Failed to initialize Ray: {e}")
        sys.exit(1)
    
    # Configure deployment
    print(f"\n📦 Deploying RISCVRISCLLMDeployment...")
    print(f"   Replicas: {args.replicas}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Route: /generate")
    
    # Create deployment with specified number of replicas
    deployment = RISCVRISCLLMDeployment.options(
        num_replicas=args.replicas,
        ray_actor_options={"num_cpus": 1}
    )
    
    # Deploy
    try:
        # Ray Serve 2.53.0: Start serve with HTTP config
        from ray.serve.config import HTTPOptions
        
        http_config = HTTPOptions(host=args.host, port=args.port)
        serve.start(http_options=http_config)
        
        # Bind deployment
        app = deployment.bind()
        
        # Run with route prefix
        serve.run(app, route_prefix="/generate", blocking=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        serve.shutdown()
        ray.shutdown()
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        print(f"\n❌ Deployment failed: {e}")
        serve.shutdown()
        ray.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
