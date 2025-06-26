import sys
import click
from know.project import init_project
from know.settings import ProjectSettings, EmbeddingSettings

@click.command()
@click.option("--project-path",
              type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=str),
              required=True,                     # ← make the argument mandatory
              help="Path to the project directory (required).")
@click.option("--project-id",
              type=str,
              help="Existing project identifier (optional).")
@click.option("--embeddings/--no-embeddings",
              "embeddings_enabled",
              default=False,
              help="Enable/disable generation of embeddings.")
@click.option("--embedding-calculator-type", default="local",
              help="Calculator type to use when embeddings are enabled.")
@click.option("--embedding-model-name", default="all-MiniLM-L6-v2",
              help="HuggingFace model name / path.")
@click.option("--embedding-normalize/--embedding-no-normalize",
              "normalize_embeddings",
              default=True,
              help="L2-normalise vectors returned by the model.")
@click.option("--embedding-device", default=None,
              help="Device to run the embedding model on (cpu / cuda:0 …).")
@click.option("--embedding-batch-size", default=32, type=int,
              help="Batch size for embedding generation.")
@click.option("--embedding-quantize/--embedding-no-quantize",
              "quantize",
              default=False,
              help="Enable int8 quantisation.")
@click.option("--embedding-quantize-bits", default=8, type=int,
              help="Number of bits when quantising the model.")
@click.option("--repo-backend",
              type=click.Choice(["memory", "duckdb"]),
              default="memory",
              help="Storage backend to use.")
@click.option("--repo-connection",
              type=str,
              default=None,
              help="Connection string / DB path for the backend.")
def cli(project_path,
        project_id,
        embeddings_enabled,
        embedding_calculator_type,
        embedding_model_name,
        normalize_embeddings,
        embedding_device,
        embedding_batch_size,
        quantize,
        embedding_quantize_bits,
        repo_backend,
        repo_connection):
    """
    Initialise a *know* project using command-line supplied settings.
    """
    embedding_cfg = EmbeddingSettings(
        calculator_type   = embedding_calculator_type,
        model_name        = embedding_model_name,
        normalize_embeddings = normalize_embeddings,
        device            = embedding_device,
        batch_size        = embedding_batch_size,
        quantize          = quantize,
        quantize_bits     = embedding_quantize_bits,
        enabled           = embeddings_enabled,
    )

    settings = ProjectSettings(
        project_path = project_path,
        project_id   = project_id,
        embedding    = embedding_cfg,
        repository_backend  = repo_backend,
        repository_connection = repo_connection,
    )

    try:
        project = init_project(settings)
        click.echo(f"Project initialised. Repo-ID: {project.get_repo().id}")
    except Exception as exc:
        click.echo(f"Project initialisation failed: {exc}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()   # noqa: E305
