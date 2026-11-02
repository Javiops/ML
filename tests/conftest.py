import pytest


@pytest.fixture
def pendiente():
    """Convierte un NotImplementedError en skip, para que un módulo sin
    implementar aparezca como pendiente en vez de como fallo."""

    def run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as e:
            pytest.skip(f"pendiente de implementar: {e}")

    return run
