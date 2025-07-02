import pytest
from pathlib import Path
from types import SimpleNamespace
from devtools import pprint

from know.lang.golang import GolangCodeParser
from know.models import ProgrammingLanguage, SymbolKind


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _make_dummy_project(root_dir: Path):
    """
    Very small stub fulfilling the interface required by GolangCodeParser.parse().
    The parser only relies on ``project_path``.
    """
    return SimpleNamespace(project_path=str(root_dir))


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_golang_parser_on_sample_file():
    """
    Parse the sample `main.go` file and assert the most important artefacts
    (imports, symbols, doc-comments…) are extracted correctly.
    """
    samples_dir = Path(__file__).parent / "samples"
    project     = _make_dummy_project(samples_dir)
    parser      = GolangCodeParser()

    parsed_file = parser.parse(project, "main.go")

    # ------------------------------------------------------------------ #
    # Basic assertions                                                    #
    # ------------------------------------------------------------------ #
    assert parsed_file.path == "main.go"
    assert parsed_file.language == ProgrammingLanguage.GO

    #pprint(parsed_file)

    # ------------------------------------------------------------------ #
    # Imports                                                             #
    # ------------------------------------------------------------------ #
    # main.go contains exactly one import:  "fmt"
    assert len(parsed_file.imports) == 2
    fmt_imp = parsed_file.imports[0]
    assert fmt_imp.virtual_path == "example.com/m"
    assert fmt_imp.external is False
    assert fmt_imp.dot is False
    assert fmt_imp.alias == "k"
    assert fmt_imp.physical_path == "."

    # ------------------------------------------------------------------ #
    # Top-level symbols                                                   #
    # ------------------------------------------------------------------ #
    symbols = {sym.name: sym for sym in parsed_file.symbols}

    # Constant
    assert "A" in symbols
    assert symbols["A"].kind == SymbolKind.CONSTANT

    # Struct with children
    assert "S" in symbols
    struct_s = symbols["S"]
    assert struct_s.kind == SymbolKind.CLASS
    assert {c.name for c in struct_s.children} == {"a", "b", "c"}

    # Method `m` attached to S (registered as top-level method symbol)
    assert "m" in symbols
    assert symbols["m"].kind == SymbolKind.METHOD

    # Function `main` with preceding comment as docstring
    assert "main" in symbols
    assert symbols["main"].kind == SymbolKind.FUNCTION
    assert symbols["main"].docstring is not None
    assert "Test comment" in symbols["main"].docstring
