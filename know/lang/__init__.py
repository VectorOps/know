from .python import PythonCodeParser, PythonLanguageHelper
from .golang import GolangCodeParser, GolangLanguageHelper
from .typescript import TypeScriptCodeParser, TypeScriptLanguageHelper
from know.parsers import CodeParserRegistry
from know.models import ProgrammingLanguage


def register_parsers():
    # TODO: Dynamic language parser discovery?
    CodeParserRegistry.register_parser(".py", PythonCodeParser)
    CodeParserRegistry.register_helper(ProgrammingLanguage.PYTHON, PythonLanguageHelper())

    CodeParserRegistry.register_parser(".go", GolangCodeParser)
    CodeParserRegistry.register_helper(ProgrammingLanguage.GO, GolangLanguageHelper())

    CodeParserRegistry.register_parser(".ts",  TypeScriptCodeParser)
    CodeParserRegistry.register_parser(".tsx", TypeScriptCodeParser)
    CodeParserRegistry.register_helper(ProgrammingLanguage.TYPESCRIPT,
                                       TypeScriptLanguageHelper())
from . import javascript   # noqa: F401  (forces registration)
