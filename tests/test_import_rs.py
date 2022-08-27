
def test_import():
    from schmeud import _schmeud
    from schmeud._schmeud import dynamics  # type: ignore
    from schmeud._schmeud import statics  # type: ignore
    from schmeud._schmeud import ml  # type: ignore
