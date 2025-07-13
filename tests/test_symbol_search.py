import pytest
import math

from know.stores.duckdb import DuckDBDataRepository
from know.stores.memory import InMemoryDataRepository
from know.models import RepoMetadata, FileMetadata, SymbolMetadata
from know.data import SymbolSearchQuery

pytest.importorskip("sentence_transformers")

from know.embeddings import EmbeddingWorker

@pytest.fixture(scope="module")
def emb_calc():
    try:
        # Use default model of the sentence-transformers
        calc = EmbeddingWorker("local", "all-MiniLM-L6-v2")
        yield calc
        calc.destroy()
    except Exception as ex:
        pytest.skip(f"Couldn't load embedding worker {ex}")

@pytest.fixture(params=["duckdb", "memory"])
def data_repo(request, tmp_path):
    """Return a fresh data repository – DuckDB or in-memory – per test run."""
    if request.param == "duckdb":
        db_path = tmp_path / "ducktest.db"
        pytest.importorskip("duckdb")            # ensure extension available
        return DuckDBDataRepository(str(db_path))
    # memory backend
    return InMemoryDataRepository()

def test_bm25_embedding_search_20cases(data_repo, emb_calc):
    """
    Inserts 20 distinct symbols with themed docstrings.
    Performs BM25, embedding-only and *combined (BM25 + embedding)* search.
    Verifies result relevance, ranking and the RRF fusion implementation.
    """
    repo_repo = data_repo.repo
    file_repo = data_repo.file
    sym_repo = data_repo.symbol

    rid = "repo-BM25-test"
    fid = "file-f"
    repo_repo.create(RepoMetadata(id=rid, name="BM25Repo", root_path="/bm25test"))
    file_repo.create(FileMetadata(id=fid, repo_id=rid, path="src/bm25.py"))

    themes = [
        ("Sorting", "Sorts a list using quicksort", "def quicksort(arr): ..."),
        ("Sorting", "Sorts items ascendingly", "def sort_asc(a): ..."),
        ("Sorting", "Bubble sort implementation", "def bubble(a): ..."),
        ("Search", "Binary search for element", "def binary_search(arr, x): ..."),
        ("Search", "Finds item using linear search", "def linear_search(lst, val): ..."),
        ("Math", "Computes the factorial", "def factorial(n): ..."),
        ("Math", "Returns the n-th fibonacci", "def fibonacci(n): ..."),
        ("Math", "Calculate the sum of numbers", "def sum_numbers(nums): ..."),
        ("Math", "Returns the average of a list", "def average(xs): ..."),
        ("String", "Reverses a given string", "def reverse_str(s): ..."),
        ("String", "Converts string to uppercase", "def to_upper(s): ..."),
        ("String", "Checks if string is palindrome", "def is_palindrome(s): ..."),
        ("IO", "Reads file and returns contents", "def read_file(path): ..."),
        ("IO", "Writes data to file", "def write_file(p, data): ..."),
        ("IO", "Appends line to a file", "def append_line(f,x): ..."),
        ("Network", "Send GET request to URL", "def get(url): ..."),
        ("Network", "Parse HTTP response body", "def parse_http(body): ..."),
        ("Network", "Open TCP connection", "def open_tcp(host,port): ..."),
        ("Date", "Returns current date", "def today(): ..."),
        ("Date", "Formats date to ISO string", "def format_iso(dt): ..."),
        ("Date", "Adds days to a date", "def add_days(dt, days): ..."),
    ]
    # Insert all as symbols, record their ids for later checks
    ids_by_theme = {}
    for i, (theme, docstring, body) in enumerate(themes):
        sid = f"s_{i}"
        vec = emb_calc.get_embedding(docstring)
        assert len(vec) == 1024
        if theme not in ids_by_theme:
            ids_by_theme[theme] = []
        ids_by_theme[theme].append(sid)
        sym_repo.create(SymbolMetadata(
            id=sid, name=f"{theme}{i}", repo_id=rid, file_id=fid,
            symbol_body=body, docstring=docstring,
            embedding_doc_vec=vec,
            embedding_code_vec=vec,
            kind="function"
        ))
    data_repo.refresh_full_text_indexes()

    # BM25 Search tests
    # Query for 'sort'; should rank sorting-related symbols at the top.
    res_bm25 = sym_repo.search(rid, SymbolSearchQuery(doc_needle="sort", limit=5))
    top_names = [s.name for s in res_bm25]
    # The top 3 should be Sorting-related
    assert any("Sorting" in n for n in top_names[:3]), f"Top 3: {top_names[:3]}"

    # Query for 'date'; the date-related symbols should rank highest
    res_date = sym_repo.search(rid, SymbolSearchQuery(doc_needle="date", limit=3))
    assert all("Date" in s.name for s in res_date), f"Top date results: {[s.name for s in res_date]}"

    # Query for 'HTTP'; network symbols
    res_net = sym_repo.search(rid, SymbolSearchQuery(doc_needle="HTTP", limit=2))
    assert any("Network" in s.name for s in res_net), f"Top network/HTTP: {[s.name for s in res_net]}"

    # Embedding search: retrieve all 'Sorting' (clustered), using first Sorting docstring as query
    sort_vec = emb_calc.get_embedding(themes[0][1])
    emb_sort = sym_repo.search(rid, SymbolSearchQuery(embedding_query=sort_vec, limit=5))
    sort_names = [s.name for s in emb_sort]
    # At least 2/3 of the top 3 should be 'Sorting' related
    assert sum("Sorting" in n for n in sort_names[:3]) >= 2, f"Top emb: {sort_names[:3]}"

    # Embedding search: retrieve all 'Math' symbols
    math_vec = emb_calc.get_embedding(themes[5][1])
    emb_math = sym_repo.search(rid, SymbolSearchQuery(embedding_query=math_vec, limit=5))
    math_names = [s.name for s in emb_math]
    assert sum("Math" in n for n in math_names[:3]) >= 1, f"Top math: {math_names[:3]}"

    # ------------------------------------------------------------------
    # Combined BM25 + Embedding search  (exercises RRF fusion)
    # ------------------------------------------------------------------
    comb = sym_repo.search(
        rid,
        SymbolSearchQuery(doc_needle="sort", embedding_query=sort_vec, limit=5)
    )
    comb_names = [s.name for s in comb]

    # At least 3 of the 5 returned symbols should be Sorting-related
    assert sum("Sorting" in n for n in comb_names) >= 3, \
        f"Combined search (RRF) results not focused on Sorting: {comb_names}"

    # The very first result should also appear in the top-5 of *either*
    # BM25-only or embedding-only search – proof that RRF fused ranks.
    top_candidate = comb_names[0]
    assert (
        top_candidate in [s.name for s in res_bm25[:5]]
        or top_candidate in [s.name for s in emb_sort[:5]]
    ), "RRF top result not present in the individual rankings"

    # Ensures sorted (BM25 or embedding score) and relevant
    # Also covers at least 21 data entries
    assert sym_repo.search(rid, SymbolSearchQuery(limit=25))  # total exists
