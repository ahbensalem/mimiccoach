"""Module-level smoke tests for the Modal app."""
from __future__ import annotations


def test_app_module_imports() -> None:
    import app  # noqa: F401


def test_modal_app_named() -> None:
    import app

    assert app.app.name == "mimiccoach"


def test_pipeline_glue_exposed() -> None:
    """analyze_from_landmarks() is the testable handle into the pipeline.
    fastapi_app is wrapped by @modal.asgi_app so we just verify it's defined."""
    import app

    assert callable(app.analyze_from_landmarks)
    assert hasattr(app, "fastapi_app")
