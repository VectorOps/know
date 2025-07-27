from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from pydantic import Field, AliasChoices, AnyHttpUrl
from pydantic_settings import SettingsConfigDict

from fastmcp.server import AccessToken, AuthSettings, Context, FastMCP, TokenVerifier

from know.project import Project, init_project
from know.settings import ProjectSettings, print_help
from know.tools.base import ToolRegistry

# Import tool modules to ensure they are registered with the ToolRegistry
from know.tools import filelist  # noqa: F401
from know.tools import filesummary  # noqa: F401
from know.tools import repomap  # noqa: F401
from know.tools import symbolsearch  # noqa: F401


@dataclass
class AppContext:
    """Application context with typed dependencies."""

    project: Project


class StaticTokenVerifier(TokenVerifier):
    """Simple token verifier that checks against a static token."""
    def __init__(self, valid_token: str):
        self._valid_token = valid_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token == self._valid_token:
            return AccessToken(sub="know-user")  # dummy subject
        return None


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


try:
    settings = Settings()
except Exception as e:
    print(f"Error: Invalid settings.\n{e}", file=sys.stderr)
    sys.exit(1)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    # Initialize on startup
    project = init_project(settings)
    try:
        yield AppContext(project=project)
    finally:
        # Cleanup on shutdown (if any)
        project.destroy()


def init():
    token_verifier = None
    auth_settings = None
    if settings.mcp_auth_token:
        token_verifier = StaticTokenVerifier(settings.mcp_auth_token)
        # AuthSettings is for RFC 9728 Protected Resource Metadata.
        auth_settings = AuthSettings(
            # Using a placeholder for issuer URL as we're using simple static token auth.
            issuer_url=AnyHttpUrl("https://auth.example.com"),
            resource_server_url=AnyHttpUrl(f"http://{settings.mcp_host}:{settings.mcp_port}"),
        )

    # authentication token can be set via --mcp-auth-token or KNOW_MCP_AUTH_TOKEN
    mcp = FastMCP(
        "Know MCP Server",
        lifespan=app_lifespan,
        token_verifier=token_verifier,
        auth=auth_settings,
    )

    # register all enabled tools with the MCP server
    tools = ToolRegistry.get_enabled_tools(settings)

    def create_handler(tool_instance):
        def handler(ctx: Context, **kwargs):
            project = ctx.request_context.lifespan_context.project
            return tool_instance.execute(project, **kwargs)
        return handler

    for tool in tools:
        schema = tool.get_openai_schema()
        # The tool's execute method has `project` as its first argument.
        # We create a handler that receives the lifespan context from the
        # MCP server and passes the project to the tool.
        mcp.add_tool(schema=schema, func=create_handler(tool))

    return mcp


mcp = init()

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port)
