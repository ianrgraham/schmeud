
def test_import():
    from schmeud import _schmeud # type: ignore
    from schmeud._schmeud import dynamics  # type: ignore
    from schmeud._schmeud import statics  # type: ignore
    from schmeud._schmeud import ml  # type: ignore
    from schmeud._schmeud import locality  # type: ignore
    from schmeud._schmeud import nlist  # type: ignore
    from schmeud._schmeud import boxdim  # type: ignore
