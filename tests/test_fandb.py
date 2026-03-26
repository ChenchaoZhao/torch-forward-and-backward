import fandb


def test_import_fandb() -> None:
    """Test that fandb module can be imported successfully."""
    assert fandb is not None


def test_fandb_has_expected_attributes() -> None:
    """Test that fandb module has expected attributes."""
    assert hasattr(fandb, "__name__")
