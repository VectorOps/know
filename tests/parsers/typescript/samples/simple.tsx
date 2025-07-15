import React from "react";

export function fn(a: number): number {
  return a + 1;
}

export class Test {
  value: number = 0;

  method(): void {
    console.log(this.value);
  }
}

const CONST = 42;
