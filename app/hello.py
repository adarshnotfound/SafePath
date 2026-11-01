"""SafePath Web Application & API Module.

Responsible for Streamlit / React frontend UI and FastAPI backend services,
rendering maps, hazard toggles, personal safety switches, and route comparisons. (Owner D)
"""

from pathlib import Path


def hello_app() -> str:
    """Print application module readiness."""
    module_name = "App & UI Module (Owner D)"
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    routes_exist = (sample_dir / "routes.json").exists()
    return f"[{module_name}] Ready. UI services configured (Sample routes available: {routes_exist})."


def main():
    print(hello_app())


if __name__ == "__main__":
    main()

