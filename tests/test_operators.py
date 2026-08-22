import numpy as np

from infcontrol.operators import (
    Formula,
    FormulaEqual,
    ImplicationChainLength,
    NumOperators,
    NumVariables,
    OperatorHistogram,
    RootOperator,
    SharedOperators,
    SharedSymbols,
    SharedVariables,
    SymbolHistogram,
    TreeDepth,
    TreeSize,
)


def test_formula_caches_structure() -> None:
    f = Formula(["->", ["→", "θ", "τ"], ["→", "ψ", "θ"]])

    assert f.size == 10
    assert f.depth == 2
    assert f.variables == {"v0", "v1", "v2"}
    assert f.operators == {"->", "→"}
    assert f.root_operator == "->"
    assert f.operator_histogram == {"->": 1, "→": 2}
    assert f.symbol_histogram["v0"] == 2
    assert f.implication_chain_length == 2


def test_unary_operators() -> None:
    f = Formula(["->", ["→", "θ", "τ"], ["→", "ψ", "θ"]])

    assert TreeSize()._call(f) == 10
    assert TreeDepth()._call(f) == 2
    assert NumVariables()._call(f) == 3
    assert NumOperators()._call(f) == 3
    assert RootOperator()._call(f) == "->"
    assert OperatorHistogram()._call(f) == {"->": 1, "→": 2}
    assert SymbolHistogram()._call(f)["v0"] == 2
    assert ImplicationChainLength()._call(f) == 2


def test_binary_operators() -> None:
    f1 = Formula(["->", ["→", "θ", "τ"], ["→", "ψ", "θ"]])
    f2 = Formula(["->", ["→", "α", "β"], ["→", "γ", "α"]])
    f3 = Formula(["->", ["→", "α", "β"], "γ"])

    assert FormulaEqual()._call(f1, f2)
    assert not FormulaEqual()._call(f1, f3)
    assert SharedVariables()._call(f1, f2) == 3
    assert SharedOperators()._call(f1, f2) == 2
    assert SharedSymbols()._call(f1, f2) == 5


def test_vectorized_unary_operators() -> None:
    f1 = Formula(["->", ["→", "θ", "τ"], ["→", "ψ", "θ"]])
    f2 = Formula(["->", ["→", "α", "β"], "γ"])
    batch = np.array([f1, f2], dtype=object)

    assert np.array_equal(TreeSize()._call(batch), np.array([10, 7]))
    assert np.array_equal(TreeDepth()._call(batch), np.array([2, 2]))
    assert np.array_equal(NumVariables()._call(batch), np.array([3, 3]))
    assert np.array_equal(NumOperators()._call(batch), np.array([3, 2]))
    assert np.array_equal(RootOperator()._call(batch), np.array(["->", "->"]))
    assert np.array_equal(ImplicationChainLength()._call(batch), np.array([2, 1]))


def test_vectorized_binary_operators_pairwise_and_scalar_broadcast() -> None:
    f1 = Formula(["->", ["→", "θ", "τ"], ["→", "ψ", "θ"]])
    f2 = Formula(["->", ["→", "α", "β"], ["→", "γ", "α"]])
    f3 = Formula(["->", ["→", "α", "β"], "γ"])
    left = np.array([f1, f1], dtype=object)
    right = np.array([f2, f3], dtype=object)

    pair_eq = FormulaEqual()._call(left, right)
    pair_vars = SharedVariables()._call(left, right)
    pair_ops = SharedOperators()._call(left, right)
    pair_symbols = SharedSymbols()._call(left, right)

    assert np.array_equal(pair_eq, np.array([True, False]))
    assert np.array_equal(pair_vars, np.array([3, 3]))
    assert np.array_equal(pair_ops, np.array([2, 2]))
    assert np.array_equal(pair_symbols, np.array([5, 5]))

    scalar_eq = FormulaEqual()._call(f1, right)
    scalar_vars = SharedVariables()._call(right, f1)
    assert np.array_equal(scalar_eq, np.array([True, False]))
    assert np.array_equal(scalar_vars, np.array([3, 3]))
