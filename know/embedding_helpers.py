from typing import TYPE_CHECKING

from know.data import SymbolFilter
from know.logger import logger
from know.models import Vector

if TYPE_CHECKING:
    from know.project import Project


# ----------------------------------------------------------------------
# Embedding helpers
# ----------------------------------------------------------------------
def schedule_symbol_embedding(symbol_repo, emb_calc, sym_id: str, body: str, sync: bool = False) -> None:
    def _on_vec(vec: Vector) -> None:
        try:
            symbol_repo.update(
                sym_id,
                {
                    "embedding_code_vec": vec,
                    "embedding_model": emb_calc.get_model_name(),
                },
            )
        except Exception as exc:                            # pragma: no cover
            logger.error(
                f"Failed to update embedding for symbol {sym_id}: {exc}",
                exc_info=True,
            )

    if sync:
        _on_vec(emb_calc.get_embedding(body))
        return

    # normal-priority request
    emb_calc.get_embedding_callback(body, _on_vec)


def schedule_missing_embeddings(project: "Project") -> None:
    """Enqueue embeddings for all symbols that still lack a vector."""
    emb_calc = project.embeddings
    if not emb_calc:
        return
    symbol_repo = project.data_repository.symbol
    repo_id     = project.get_repo().id
    PAGE_SIZE = 1_000
    offset = 0
    while True:
        page = symbol_repo.get_list(
            SymbolFilter(
                repo_id=repo_id,
                has_embedding=False,
                limit=PAGE_SIZE,
                offset=offset,
            ),
        )
        if not page:
            break
        for sym in page:
            if sym.body:
                schedule_symbol_embedding(
                    symbol_repo,
                    emb_calc,
                    sym_id=sym.id,
                    body=sym.body,
                    sync=project.settings.sync_embeddings,
                )
        offset += PAGE_SIZE

def schedule_outdated_embeddings(project: "Project") -> None:
    """
    Re-enqueue embeddings for all symbols whose stored vector was
    generated with a *different* model than the one currently configured
    in `project.embeddings`.
    """
    emb_calc = project.embeddings
    if not emb_calc:      # embeddings disabled
        return

    model_name   = emb_calc.get_model_name()
    symbol_repo  = project.data_repository.symbol
    repo_id      = project.get_repo().id
    PAGE_SIZE    = 1_000
    offset       = 0

    # TODO: Add data filter
    while True:
        page = symbol_repo.get_list(
            SymbolFilter(
                repo_id=repo_id,
                limit=PAGE_SIZE,
                offset=offset,
            ),
        )
        if not page:
            break

        for sym in page:
            # symbol already has an embedding → but with a *different* model
            if (
                sym.body
                and sym.embedding_model
                and sym.embedding_model != model_name
            ):
                schedule_symbol_embedding(
                    symbol_repo,
                    emb_calc,
                    sym_id=sym.id,
                    body=sym.body,
                    sync=project.settings.sync_embeddings,
                )

        offset += PAGE_SIZE
