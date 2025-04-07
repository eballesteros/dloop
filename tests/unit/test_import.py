def test_import():
    """Test that the dloop package can be imported successfully."""

    # Check that main classes are imported
    # These imports are intentionally used only to verify they exist
    # fmt: off
    from dloop import Event, LoopEvents, Loop, LoopState  # noqa: F401, I001


def test_version():
    """Test that the package version is available and properly formatted."""
    import dloop

    # Check that __version__ exists
    assert hasattr(dloop, "__version__")

    # Check that version is a string
    assert isinstance(dloop.__version__, str)

    # Check that version follows semantic versioning (optional)
    import re

    assert re.match(
        r"^\d+\.\d+\.\d+(?:[-\w](?:[-\w]*[-\w])?)?(?:\+[-\w](?:[-\w]*[-\w])?)?$", dloop.__version__
    )
