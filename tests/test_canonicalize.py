from infcontrol.canonicalize import canonicalize, is_variable, num_variables, tree_depth, tree_size, variables


def test_is_variable_detects_greek_symbols() -> None:
    assert is_variable("θ")
    assert is_variable("𝜑")
    assert not is_variable("v0")
    assert not is_variable("foo")
    assert not is_variable(1)


def test_canonicalize_renames_by_first_occurrence() -> None:
    expr = ["→", "θ", ["→", "ψ", "θ"]]
    assert canonicalize(expr) == ["→", "v0", ["→", "v1", "v0"]]


def test_tree_properties_and_variable_set() -> None:
    expr = ["->", ["→", "θ", "τ"], ["→", "ψ", "θ"]]
    assert tree_size(expr) == 10
    assert tree_depth(expr) == 2
    assert variables(expr) == {"θ", "τ", "ψ"}
    assert num_variables(expr) == 3
