import React from "react";

export function fn(a: number): number {
  return a + 1;
}

// Hello World
export class Test extends Foo {
  value: number = 0;

  method(): void {
    console.log(this.value);
  }
}

const CONST = 42;
let z = "foobar";
export {z};
