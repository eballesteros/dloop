def test_import():
    """Test that the dloop package can be imported successfully."""
    import dloop
    
    # Check that main classes are imported
    from dloop import Event, Loop, LoopEvents