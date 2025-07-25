from __future__ import annotations

from typing import Type

from pydantic import BaseModel

from know.settings import iter_settings


def print_help(model: Type[BaseModel], script_name: str, kebab: bool = True):
    """
    Print a formatted help message for a Pydantic settings model.

    Parameters
    ----------
    model : BaseModel | BaseSettings subclass
    script_name : file name to show in the "usage: " line
    kebab : convert snake_case to kebab‑case for CLI flags
    """
    print(f"usage: {script_name} [OPTIONS]")
    print("\nOptions:")
    for opt in iter_settings(model, kebab=kebab):
        flags = [opt.flag] + opt.aliases
        flag_str = ", ".join(flags)
        line = f"  {flag_str:<40} {opt.description}"

        details = []
        if opt.is_required:
            details.append("required")

        if opt.default_value is not ...:
            # for multiline defaults, only show the first line
            default_str = str(opt.default_value).split("\n")[0]
            details.append(f"default: {default_str!r}")

        if details:
            line += f" [{', '.join(details)}]"
        print(line)
