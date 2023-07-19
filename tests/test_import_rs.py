def test_rust_import():
    """Test that Rust extensions can be imported."""
    from schmeud import _schmeud  # noqa: F401
    from schmeud._schmeud import dynamics  # noqa: F401
    from schmeud._schmeud import statics  # noqa: F401
    from schmeud._schmeud import ml  # noqa: F401
    from schmeud._schmeud import locality  # noqa: F401
    from schmeud._schmeud import nlist  # noqa: F401
    from schmeud._schmeud import boxdim  # noqa: F401
