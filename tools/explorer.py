import sys
import faulthandler
import signal
import pathlib
from flask import Flask, render_template, url_for, redirect, abort
from markupsafe import Markup
from flask import request
from pydantic import Field, AliasChoices
from pydantic_settings import SettingsConfigDict
from know.settings import ProjectSettings, EmbeddingSettings, print_help
from know.project  import init_project, Project
from know.tools.repomap import RepoMapTool
from know.tools.nodesearch import NodeSearchTool
from know.models import NodeKind, Visibility
from know.data     import (
    AbstractDataRepository, RepoMetadata, PackageMetadata, FileMetadata,
    Node, ImportEdge, NodeRef, PackageFilter, FileFilter, NodeFilter, ImportEdgeFilter, NodeRefFilter
)
from know.file_summary import SummaryMode, build_file_summary

import logging
logging.basicConfig(level=logging.DEBUG)

class Settings(ProjectSettings):
    """Explorer-specific settings, extending project settings."""
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_enforce_required=True,
        env_prefix="KNOW_",
        env_nested_delimiter="_",
    )

    project_path: str = Field(
        description="Root directory of the project to analyse/assist with.",
        validation_alias=AliasChoices("project-path", "p", "path"),
    )

    host: str = Field("127.0.0.1", description="Host to bind the web server to.")
    port: int = Field(5000, description="Port to bind the web server to.")
    debug: bool = Field(False, description="Enable Flask debug mode.")


def create_app(project) -> Flask:
    template_folder = str(pathlib.Path(__file__).with_suffix('').parent / "templates")
    app = Flask(__name__, template_folder=template_folder)
    data = project.data_repository        # shortcut to the AbstractDataRepository

    @app.route("/")
    def index():
        return redirect(url_for("list_repos"))

    # ----- repo -----
    @app.route("/repos")
    def list_repos():
        items = data.repo.get_list_by_ids([])
        return render_template("explorer/list.html", title="Repositories",
                               items=items, item_type="repo")

    @app.route("/repos/<repo_id>")
    def repo_detail(repo_id):
        repo = data.repo.get_by_id(repo_id) or abort(404)
        pkgs = data.package.get_list(PackageFilter(repo_id=repo_id))
        files = data.file.get_list(FileFilter(repo_id=repo_id))
        return render_template("explorer/repo_detail.html",
                               item=repo, packages=pkgs, files=files)

    # ----- package -----
    @app.route("/packages")
    def list_packages():
        items = data.package.get_list(PackageFilter())
        return render_template("explorer/list.html", title="Packages",
                               items=items, item_type="package")

    @app.route("/packages/<package_id>")
    def package_detail(package_id):
        pkg = data.package.get_by_id(package_id) or abort(404)
        files = data.file.get_list(FileFilter(package_id=package_id))
        symbols = data.symbol.get_list(NodeFilter(package_id=package_id))
        importedges  = data.importedge.get_list(
            ImportEdgeFilter(source_package_id=package_id)
        )
        return render_template("explorer/detail_generic.html",
                               item=pkg, files=files, symbols=symbols, importedges=importedges)

    # ----- file -----
    @app.route("/files")
    def list_files():
        items = data.file.get_list(FileFilter())
        return render_template("explorer/list.html", title="Files",
                               items=items, item_type="file")

    @app.route("/files/<file_id>")
    def file_detail(file_id):
        file = data.file.get_by_id(file_id) or abort(404)

        symbols      = data.symbol.get_list(NodeFilter(file_id=file_id))
        importedges  = data.importedge.get_list(
            ImportEdgeFilter(source_file_id=file_id)
        )
        file_summary_obj = build_file_summary(project, file.path, summary_mode=SummaryMode.ShortSummary)
        summary = file_summary_obj.content if file_summary_obj else "Could not generate summary."

        return render_template(
            "explorer/detail_generic.html",
            item=file,
            symbols=symbols,
            importedges=importedges,
            summary=summary,
        )

    # ----- symbol -----
    @app.route("/symbols/<symbol_id>")
    def symbol_detail(symbol_id):
        symbol = data.symbol.get_by_id(symbol_id) or abort(404)
        return render_template("explorer/detail_symbol.html",
                               item=symbol)

    # ----- importedge -----
    @app.route("/importedges")
    def list_importedges():
        items = data.importedge.get_list(ImportEdgeFilter())
        return render_template("explorer/list.html", title="Import Edges",
                               items=items, item_type="importedge")

    @app.route("/importedges/<importedge_id>")
    def importedge_detail(importedge_id):
        edge = data.importedge.get_by_id(importedge_id) or abort(404)
        return render_template("explorer/detail_generic.html",
                               item=edge)

    # ----- symbolref -----
    @app.route("/symbolrefs")
    def list_symbolrefs():
        items = data.symbolref.get_list(NodeRefFilter())
        return render_template("explorer/list.html", title="Symbol Refs",
                               items=items, item_type="symbolref")

    @app.route("/symbolrefs/<symbolref_id>")
    def symbolref_detail(symbolref_id):
        ref = data.symbolref.get_by_id(symbolref_id) or abort(404)
        return render_template("explorer/detail_generic.html",
                               item=ref)

    @app.route("/tools/repomap", methods=["GET", "POST"])
    def repomap_tool():
        results = None
        form_data = request.form

        if request.method == "POST":
            symbol_names = [s.strip() for s in form_data.get("symbol_names", "").split(",") if s.strip()]
            file_paths = [p.strip() for p in form_data.get("file_paths", "").split(",") if p.strip()]
            prompt = form_data.get("prompt") or None

            limit = int(form_data.get("limit") or 20)
            restart_prob = float(form_data.get("restart_prob") or 0.15)
            min_symbol_len = int(form_data.get("min_symbol_len") or 3)
            summary_mode = SummaryMode(form_data.get("summary_mode", SummaryMode.ShortSummary.value))
            
            token_limit_count_str = form_data.get("token_limit_count")
            token_limit_count = int(token_limit_count_str) if token_limit_count_str else None
            token_limit_model = form_data.get("token_limit_model") or None

            tool = RepoMapTool()
            raw_results = tool.execute(
                project,
                tool.tool_input(
                    symbol_names=symbol_names,
                    file_paths=file_paths,
                    prompt=prompt,
                    limit=limit,
                    restart_prob=restart_prob,
                    summary_mode=summary_mode,
                    min_symbol_len=min_symbol_len,
                    token_limit_count=token_limit_count,
                    token_limit_model=token_limit_model,
                )
            )
            
            enriched_results = []
            for r in raw_results:
                file_obj = data.file.get_by_path(r.file_path)
                enriched_results.append({
                    "file_path": r.file_path,
                    "file_obj": file_obj,
                    "score": r.score,
                    "summary": r.summary,
                })
            results = enriched_results

        return render_template("explorer/repomap.html",
                               title="RepoMap Tool",
                               summary_modes=[e.value for e in SummaryMode],
                               results=results,
                               form_values=form_data)

    @app.route("/tools/symbolsearch", methods=["GET", "POST"])
    def symbolsearch_tool():
        results = None
        form_data = request.form

        if request.method == "POST":
            tool = SearchSymbolsTool()
            raw_results = tool.execute(
                project,
                tool.tool_input(
                    symbol_name=form_data.get("symbol_name") or None,
                    symbol_fqn=form_data.get("symbol_fqn") or None,
                    kind=form_data.get("kind") or None,
                    visibility=form_data.get("visibility") or "public",
                    query=form_data.get("query") or None,
                    limit=int(form_data.get("limit") or 20),
                    summary_mode=SummaryMode(form_data.get("summary_mode", SummaryMode.ShortSummary.value)),
                )
            )

            enriched_results = []
            for r in raw_results:
                if r.file_path:
                    r.file_obj = data.file.get_by_path(r.file_path)
                else:
                    r.file_obj = None
                enriched_results.append(r)
            results = enriched_results

        return render_template("explorer/symbolsearch.html",
                               title="Symbol Search Tool",
                               results=results,
                               form_values=form_data,
                               node_kinds=[k.value for k in NodeKind],
                               visibilities=[v.value for v in Visibility] + ["all"],
                               summary_modes=[e.value for e in SummaryMode])

    def _link_to(obj):
        mapping = {RepoMetadata:"repo", PackageMetadata:"package", FileMetadata:"file",
                   Node:"symbol", ImportEdge:"importedge", NodeRef:"symbolref"}
        return url_for(f"{mapping[type(obj)]}_detail", **{f"{mapping[type(obj)]}_id": obj.id})

    app.jinja_env.globals["_link_to"] = _link_to
    # expose getattr so templates can safely access attributes with defaults
    app.jinja_env.globals["getattr"] = getattr

    def _linkify(value, field_name):
        """
        Turn *_id fields into HTML links pointing at the referenced object.
        Used as a Jinja filter.
        """
        mapping = {
            "repo_id":           "repo",
            "package_id":        "package",
            "file_id":           "file",
            "symbol_id":         "symbol",
            "parent_node_id":    "symbol",
            "from_package_id":   "package",
            "from_file_id":      "file",
            "to_package_id":     "package",
        }
        target = mapping.get(field_name)
        if target and isinstance(value, str) and value:
            url = url_for(f"{target}_detail", **{f"{target}_id": value})
            return Markup(f'<a href="{url}">{value}</a>')
        return value

    app.jinja_env.filters["linkify"] = _linkify    # NEW

    return app

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help(Settings, "explorer.py")
        sys.exit(0)

    faulthandler.register(signal.SIGHUP)

    try:
        settings = Settings()
    except Exception as e:
        print(f"Error: Invalid settings.\n{e}", file=sys.stderr)
        sys.exit(1)

    project = init_project(settings)
    app = create_app(project)
    app.run(host=settings.host, port=settings.port, debug=settings.debug)

if __name__ == "__main__":
    main()
