import sys
import inspect
from dataclasses import asdict
from typing import TYPE_CHECKING
from pydantic import Field, AliasChoices, AnyHttpUrl
from pydantic_settings import SettingsConfigDict


from know import Project, init_project
from know.settings import ProjectSettings
from know.tools.base import ToolRegistry

if TYPE_CHECKING:
    from fastmcp import FastMCP, Context  # type: ignore[import-not-found]
    from fastmcp.server.dependencies import get_context  # type: ignore[import-not-found]

try:
    from fastmcp import FastMCP, Context
    from fastmcp.server.dependencies import get_context
except ImportError:
    print(
        "Error: `fastmcp` is not installed. Please install it with `pip install 'vectorops[fastmcp]'` to use the MCP server.",
        file=sys.stderr,
    )
    sys.exit(1)

from contextlib import asynccontextmanager


class Settings(ProjectSettings):
    model_config = SettingsConfigDict(
        env_prefix="KNOW_",
        env_nested_delimiter="_",
    )

    mcp_host: str = Field("127.0.0.1", description="MCP server host.")
    mcp_port: int = Field(8000, description="MCP server port.")
    mcp_auth_token: str | None = Field(None, description="MCP server auth token (optional).")


def create_mcp_app() -> tuple["FastMCP", Settings]:
    settings = Settings()
    mcp: FastMCP = FastMCP(
        "vectorops",
        auth_token=settings.mcp_auth_token,
    )
    project = init_project(settings)

    # register all enabled tools with the MCP server
    tools = ToolRegistry.get_enabled_tools(settings)
    for tool in tools:
        mcp_def = asdict(tool.get_mcp_definition(project))
        fn = mcp_def.pop("fn")
        mcp.tool(fn, **mcp_def)

    return mcp, settings


mcp, settings = create_mcp_app()


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=settings.mcp_host,
        port=settings.mcp_port,
    )
