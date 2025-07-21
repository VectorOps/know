import React from "react";

// CommonJS import
const circle = require('./circle.js');

// ------------------------------------------------------------------
// exported consts / arrow-fn
export const j1 = 10, f1 = () => {};

// plain declarations
let a1, b1, c1 = 10;
var e2 = 20, f;

// exported function
export function fn(a) {
  return a + 1;
}

// async arrow-function assigned to const
const a = async (b) => {
  alert("foo");
};

// ------------------------------------------------------------------
// exported class with members
export class Test extends Foo {
  value = 0;

  foo = () => { /* … */ }

  async method(b) {
    console.log(this.value);
  }
}

// plain class
class Base {
  getName() { return "Base"; }
  printName() { console.log("Hello, " + this.getName()); }
}

// misc variables / re-export
const CONST = 42;
let z = "foobar";
export { z };

const Foo = class {
  bar() {
    return 123;
  }
};

// assignment with arrow-fn
window.onload = () => { alert("yes"); };

// plain function
function identity(arg) {
  return arg;
}
