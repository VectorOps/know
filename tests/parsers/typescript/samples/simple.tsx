import React from "react";

foo = () => {};

export const j1 = 10, f1 = () => {};

let a1, b1, c1 = 10;

var e2 = 20, f;

export function fn(a: number): number {
  return a + 1;
}

const a = async (b: str) => {
  alert("foo");
};

// Hello World
export class Test extends Foo {
  value: number = 0;

  foo = () => {
    // Bar
  }

  async method(b: str): void {
    console.log(this.value);
  }
}

interface LabeledValue {
  label: string;
}

abstract class Base {
  abstract getName(): string;

  printName() {
    console.log("Hello, " + this.getName());
  }
}

type Point = {
  x: number;
  y: number;
};

enum Direction {
  Up = 1,
  Down,
  Left,
  Right,
}

const CONST = 42;
let z = "foobar";
export {z};

namespace Validation {
  export interface StringValidator {
    isAcceptable(s: string): boolean;
  }
  const lettersRegexp = /^[A-Za-z]+$/;
  const numberRegexp = /^[0-9]+$/;
  const fn = () => {
    alert("yes");
  };
}

window.onload = () => {
    alert("yes");
};
