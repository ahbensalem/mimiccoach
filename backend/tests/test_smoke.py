"""P0 smoke tests — verify the package imports and the Modal app declares /healthz."""
from __future__ import annotations


def test_app_module_imports() -> None:
    import app  # noqa: F401


def test_healthz_function_exists() -> None:
    import app

    assert hasattr(app, "healthz"), "Modal app must expose healthz"


def test_modal_app_named() -> None:
    import app

    assert app.app.name == "mimiccoach"
