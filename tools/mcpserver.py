from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
import inspect

import uvicorn
from pydantic import Field, AliasChoices, AnyHttpUrl
from pydantic_settings import SettingsConfigDict

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP, Context

from know.project import Project, init_project
from know.settings import ProjectSettings, print_help
from know.tools.base import ToolRegistry


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
        "vectorops",
        lifespan=app_lifespan,
        token_verifier=token_verifier,
        auth=auth_settings,
    )

    # register all enabled tools with the MCP server
    tools = ToolRegistry.get_enabled_tools(settings)

    def create_handler(tool_instance):
        tool_input_type = tool_instance.tool_input
        tool_output_type = tool_instance.tool_output

        def handler(ctx: Context, req):
            project = ctx.request_context.lifespan_context.project
            return tool_instance.execute(project, req)

        # Dynamically create a signature for the handler to allow FastMCP to
        # generate a correct schema for the tool's input and output.
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        params[1] = params[1].replace(annotation=tool_input_type)
        sig = sig.replace(parameters=params, return_annotation=tool_output_type)
        handler.__signature__ = sig

        return handler

    for tool in tools:
        mcp_def = tool.get_mcp_definition()
        mcp.add_tool(fn=create_handler(tool), **asdict(mcp_def))

    return mcp


mcp = init()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
