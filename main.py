import sys
import argparse
import uvicorn

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="CourtRoom AI - Legal Simulation Platform")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the API server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the API server (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reloading for backend development",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run legacy Streamlit application instead",
    )

    args = parser.parse_args()

    if args.streamlit:
        import subprocess
        print("[CourtRoom AI] Launching legacy Streamlit UI...")
        subprocess.run(["streamlit", "run", "ui/app.py"])
        return

    print(f"[CourtRoom AI] Starting API Server at http://{args.host}:{args.port}")
    print(f"[CourtRoom AI] Interactive API Documentation available at http://{args.host}:{args.port}/docs")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
