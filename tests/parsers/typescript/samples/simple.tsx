import React from "react";

export * from './foobar.ts';
export { bar } from './foobar.ts';

foo = () => {};

export const j1 = 10, f1 = () => {};

export var a = 10;

let a1, b1, c1 = 10;

var e2 = 20, f;

(function(a) {
  // foobar
}(10));

export function fn(a: number): number {
  return a + 1;
}

export type Foo = {
  z: number;
};

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

const Foo = class {
  bar() {
    return 123;
  }
};

function identity<Type>(arg: Type): Type {
  return arg;
}

let myIdentity: <Type>(arg: Type) => Type = identity;

interface GenericIdentityFn {
  <Type>(arg: Type): Type;
}

class GenericNumber<NumType> {
  zeroValue: NumType;
  add: (x: NumType, y: NumType) => NumType;
}

export default function MyApp() {
  return (
    <div>
      <h1>Welcome to my app</h1>
      <MyButton title="I'm a button" />
    </div>
  );
}

const circle = require('./circle.js');
exports.area = (r) => PI * r ** 2;

{
  function test() {
    console.log("yes");
  }
  "test"
}
(
  "text"
)
