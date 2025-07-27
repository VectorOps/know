from __future__ import annotations

import sys
from functools import partial

import uvicorn
from pydantic import Field, AliasChoices
from pydantic_settings import SettingsConfigDict

from fastmcp.server import Auth, FastMCP

from know.project import init_project
from know.settings import ProjectSettings, print_help
from know.tools.base import ToolRegistry

# Import tool modules to ensure they are registered with the ToolRegistry
from know.tools import filelist  # noqa: F401
from know.tools import filesummary  # noqa: F401
from know.tools import repomap  # noqa: F401
from know.tools import symbolsearch  # noqa: F401


class Settings(ProjectSettings):
    """MCP-Server specific settings, extending project settings."""
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

    mcp_host: str = Field("127.0.0.1", description="MCP server host.")
    mcp_port: int = Field(8000, description="MCP server port.")
    mcp_auth_token: str | None = Field(None, description="MCP server auth token (optional).")


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help(Settings, "mcpserver.py")
        sys.exit(0)

    try:
        settings = Settings()
    except Exception as e:
        print(f"Error: Invalid settings.\n{e}", file=sys.stderr)
        sys.exit(1)

    project = init_project(settings)

    # authentication token can be set via --mcp-auth-token or KNOW_MCP_AUTH_TOKEN
    mcp = FastMCP(
        auth=Auth(token=settings.mcp_auth_token) if settings.mcp_auth_token else None,
    )

    # register all enabled tools with the MCP server
    tools = ToolRegistry.get_enabled_tools(settings)
    for tool in tools:
        schema = tool.get_openai_schema()
        # The tool's execute method has `project` as its first argument.
        # We use functools.partial to curry it, so the MCP framework can
        # call the tool with only the arguments from the tool-use request.
        handler = partial(tool.execute, project)
        mcp.add_tool(schema=schema, func=handler)

    # run the server
    uvicorn.run(mcp, app_dir=".", host=settings.mcp_host, port=settings.mcp_port)


if __name__ == "__main__":
    main()
