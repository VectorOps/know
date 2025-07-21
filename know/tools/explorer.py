import argparse, pathlib
from flask import Flask, render_template, url_for, redirect, abort
from know.settings import ProjectSettings, EmbeddingSettings
from know.project  import init_project, Project
from know.data     import (
    AbstractDataRepository, RepoMetadata, PackageMetadata, FileMetadata,
    SymbolMetadata, ImportEdge, SymbolRef, PackageFilter, FileFilter, SymbolFilter, ImportEdgeFilter, SymbolRefFilter
)

def _parse_cli():
    p = argparse.ArgumentParser(prog="project-explorer",
                                description="Flask UI for browsing a project")
    p.add_argument("-p", "--path", required=True)
    p.add_argument("--repo-backend", choices=["memory", "duckdb"], default="duckdb")
    p.add_argument("--repo-connection", default=None)
    p.add_argument("--enable-embeddings", action="store_true")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--embedding-cache-backend",
                   choices=["duckdb", "sqlite", "none"], default="duckdb")
    p.add_argument("--embedding-cache-path", default="cache.duckdb")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

def create_project(args) -> Project:
    ps_kwargs = {
        "project_path":         args.path,
        "repository_backend":   args.repo_backend,
        "repository_connection": args.repo_connection,
    }
    if getattr(args, "enable_embeddings", False):
        ps_kwargs["embedding"] = EmbeddingSettings(
            enabled=True,
            model_name=args.embedding_model,
            cache_backend=args.embedding_cache_backend,
            cache_path=args.embedding_cache_path,
        )
    return init_project(ProjectSettings(**ps_kwargs))

def create_app(project) -> Flask:
    template_folder = str(pathlib.Path(__file__).with_suffix('').parent / "templates")
    app = Flask(__name__, template_folder=template_folder)

    @app.route("/")
    def index():
        return redirect(url_for("list_repos"))

    # ----- repo -----
    @app.route("/repos")
    def list_repos():
        items = project.repo.get_list_by_ids([]) or [project.repo.get_by_id(r.id) for r in project.repo.get_list_by_ids([])]
        return render_template("explorer/list.html", title="Repositories",
                               items=items, item_type="repo")

    @app.route("/repos/<repo_id>")
    def repo_detail(repo_id):
        repo = project.repo.get_by_id(repo_id) or abort(404)
        pkgs = project.package.get_list(PackageFilter(repo_id=repo_id))
        files = project.file.get_list(FileFilter(repo_id=repo_id))
        return render_template("explorer/repo_detail.html",
                               item=repo, packages=pkgs, files=files)

    # ----- package -----
    @app.route("/packages")
    def list_packages():
        items = project.package.get_list(PackageFilter())
        return render_template("explorer/list.html", title="Packages",
                               items=items, item_type="package")

    @app.route("/packages/<package_id>")
    def package_detail(package_id):
        pkg = project.package.get_by_id(package_id) or abort(404)
        files = project.file.get_list(FileFilter(package_id=package_id))
        symbols = project.symbol.get_list(SymbolFilter(package_id=package_id))
        return render_template("explorer/detail_generic.html",
                               item=pkg, files=files, symbols=symbols)

    # ----- file -----
    @app.route("/files")
    def list_files():
        items = project.file.get_list(FileFilter())
        return render_template("explorer/list.html", title="Files",
                               items=items, item_type="file")

    @app.route("/files/<file_id>")
    def file_detail(file_id):
        file = project.file.get_by_id(file_id) or abort(404)
        symbols = project.symbol.get_list(SymbolFilter(file_id=file_id))
        return render_template("explorer/detail_generic.html",
                               item=file, symbols=symbols)

    # ----- symbol -----
    @app.route("/symbols")
    def list_symbols():
        items = project.symbol.get_list(SymbolFilter())
        return render_template("explorer/list.html", title="Symbols",
                               items=items, item_type="symbol")

    @app.route("/symbols/<symbol_id>")
    def symbol_detail(symbol_id):
        symbol = project.symbol.get_by_id(symbol_id) or abort(404)
        return render_template("explorer/detail_generic.html",
                               item=symbol)

    # ----- importedge -----
    @app.route("/importedges")
    def list_importedges():
        items = project.importedge.get_list(ImportEdgeFilter())
        return render_template("explorer/list.html", title="Import Edges",
                               items=items, item_type="importedge")

    @app.route("/importedges/<importedge_id>")
    def importedge_detail(importedge_id):
        edge = project.importedge.get_by_id(importedge_id) or abort(404)
        return render_template("explorer/detail_generic.html",
                               item=edge)

    # ----- symbolref -----
    @app.route("/symbolrefs")
    def list_symbolrefs():
        items = project.symbolref.get_list(SymbolRefFilter())
        return render_template("explorer/list.html", title="Symbol Refs",
                               items=items, item_type="symbolref")

    @app.route("/symbolrefs/<symbolref_id>")
    def symbolref_detail(symbolref_id):
        ref = project.symbolref.get_by_id(symbolref_id) or abort(404)
        return render_template("explorer/detail_generic.html",
                               item=ref)

    def _link_to(obj):
        mapping = {RepoMetadata:"repo", PackageMetadata:"package", FileMetadata:"file",
                   SymbolMetadata:"symbol", ImportEdge:"importedge", SymbolRef:"symbolref"}
        return url_for(f"{mapping[type(obj)]}_detail", **{f"{mapping[type(obj)]}_id": obj.id})

    app.jinja_env.globals["_link_to"] = _link_to

    return app

def main():
    args = _parse_cli()
    project = create_project(args)
    app = create_app(project)
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
