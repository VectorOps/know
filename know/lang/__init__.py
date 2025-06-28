from .python.parser import PythonCodeParser
from know.parsers import CodeParserRegistry
from know.models import ProgrammingLanguage


def register_parsers():
    # TODO: Dynamic language parser discovery?
    PythonCodeParser.register()
