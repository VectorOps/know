from .python import PythonCodeParser, PythonLanguageHelper
from .golang import GolangCodeParser, GolangLanguageHelper
from know.parsers import CodeParserRegistry
from know.models import ProgrammingLanguage


def register_parsers():
    # TODO: Dynamic language parser discovery?
    CodeParserRegistry.register_parser(".py", PythonCodeParser)
    CodeParserRegistry.register_helper(ProgrammingLanguage.PYTHON, PythonLanguageHelper())

    CodeParserRegistry.register_parser(".go", GolangCodeParser)
    CodeParserRegistry.register_helper(ProgrammingLanguage.GO, GolangLanguageHelper())
