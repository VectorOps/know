from .python import PythonCodeParser
from .golang import GolangCodeParser
from know.parsers import CodeParserRegistry
from know.models import ProgrammingLanguage


def register_parsers():
    # TODO: Dynamic language parser discovery?
    PythonCodeParser.register()
    GolangCodeParser.register()
