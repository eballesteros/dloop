def test_import():
    """Test that the dloop package can be imported successfully."""

    # Check that main classes are imported
    # These imports are intentionally used only to verify they exist
    # fmt: off
    from dloop import Event, LoopEvents, Loop, LoopState  # noqa: F401, I001
