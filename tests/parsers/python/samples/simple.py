#!/usr/bin/env python3
from os import getcwd
import foobar
from .foobuz import abc as d

# Comment
CONST = "abc"

# Dummy function
def fn(a, b, c: str):
    "docstring!"
    return a + b + c


# Another function
def _foo(a: int):
    """
    Multiline
    Docstring
    """
    return a


@abc
def decorated(a, b):
    return a * b


@abc
@fed
def double_decorated():
    pass


# Ellipsis fn
def ellipsis_fn():
    ...


# 123
ellipsis_fn()


# Async function
@tss
async def async_fn(a, b):
    return a - b


# Class
class Test:
    ABC = "abc"

    def __init__(self):
        "constructor"
        self.a = 10

    def method(self):
        """
        Multilino
        """
        pass

    @property
    def get(self):
        return self.a

    async def async_method(self):
        pass

    @abc
    @fed
    def multi_decorated(self):
        pass

    def ellipsis_method(self, a: int) -> int:
        """
        Test me
        """
        ...


# Decorated class
@dummy
class Foobar(Foo, Bar, Buzz):
    pass


d()

if __name__ == "__main__":
    import foo
    print("123")
