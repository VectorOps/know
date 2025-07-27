from __future__ import annotations

import sys
import inspect
from dataclasses import asdict
from pydantic import Field, AliasChoices, AnyHttpUrl
from pydantic_settings import SettingsConfigDict


from know.project import Project, init_project
from know.settings import ProjectSettings
from know.tools.base import ToolRegistry

from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_context
from contextlib import asynccontextmanager


class Settings(ProjectSettings):
    """MCP-Server specific settings, extending project settings."""
    model_config = SettingsConfigDict(
        #cli_parse_args=True,
        #cli_kebab_case=True,
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
async def lifespan(server: FastMCP):
    """Manage application lifecycle with type-safe context."""
    # Initialize on startup
    project = init_project(settings)
    server.state.project = project
    try:
        yield
    finally:
        # Cleanup on shutdown (if any)
        project.destroy()


def init():
    mcp = FastMCP(
        "vectorops",
        lifespan=lifespan,
    )

    # register all enabled tools with the MCP server
    tools = ToolRegistry.get_enabled_tools(settings)

    def create_handler(tool_instance):
        tool_input_type = tool_instance.tool_input
        tool_output_type = tool_instance.tool_output

        def handler(req):
            project = get_context().fastmcp.state.project
            return tool_instance.execute(project, req)

        # Dynamically create a signature for the handler to allow FastMCP to
        # generate a correct schema for the tool's input and output.
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        params[0] = params[0].replace(annotation=tool_input_type)
        sig = sig.replace(parameters=params, return_annotation=tool_output_type)
        handler.__signature__ = sig

        return handler

    for tool in tools:
        mcp_def = tool.get_mcp_definition()
        mcp.tool(create_handler(tool), **asdict(mcp_def))

    return mcp


mcp = init()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
