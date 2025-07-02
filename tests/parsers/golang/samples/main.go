package main

import (
	k "example.com/m"
	"fmt"
)

const (
	A = 10
)

type S struct {
	a int `valid:"test"`
	b str
	c *T
}

func (s *S) m(a int) error {
	return nil
}

/*
Just a comment
*/
func dummy(a int) (int, error) {
	return a, nil
}

// Test comment
func main() {
	a := S{}
	a.m(A)
	k.foobar()
}
