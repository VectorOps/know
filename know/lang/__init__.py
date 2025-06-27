from .python.parser import PythonCodeParser
from know.parsers import CodeParserRegistry


def register_parsers():
    CodeParserRegistry.register_parser(".py", PythonCodeParser())
