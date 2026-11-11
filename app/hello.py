"""SafePath Web Application & API Module.

Responsible for Streamlit / React frontend UI and FastAPI backend services,
rendering maps, hazard toggles, personal safety switches, and route comparisons. (Owner D)
"""

from pathlib import Path


def hello_app() -> str:
    """Print application module readiness."""
    module_name = "App & UI Module (Owner D)"
    base = Path(__file__).resolve().parent.parent / "data"
    proc_routes = (base / "processed" / "routes.json").exists()
    sample_routes = (base / "sample" / "routes.json").exists()
    return (
        f"[{module_name}] Ready. Streamlit UI configured at app/main.py "
        f"(Processed routes: {proc_routes}, Sample routes: {sample_routes})."
    )


def main():
    print(hello_app())


if __name__ == "__main__":
    main()

