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
    model_config = SettingsConfigDict(
        env_prefix="KNOW_",
        env_nested_delimiter="_",
    )

    project_path: str = Field(
        description="Root directory of the project to analyze/assist with.",
    )

    mcp_host: str = Field("127.0.0.1", description="MCP server host.")
    mcp_port: int = Field(8000, description="MCP server port.")
    mcp_auth_token: str | None = Field(None, description="MCP server auth token (optional).")


mcp: FastMCP = FastMCP(
    "vectorops",
)


def init():
    settings = Settings()
    print(settings)
    project = init_project(settings)

    # register all enabled tools with the MCP server
    tools = ToolRegistry.get_enabled_tools(settings)
    for tool in tools:
        mcp_def = asdict(tool.get_mcp_definition(project))
        fn = mcp_def.pop("fn")
        mcp.tool(fn, **mcp_def)

    return mcp

init()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
