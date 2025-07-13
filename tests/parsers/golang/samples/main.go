package main

import (
	k "example.com/m"
	"fmt"
)

const (
	A = 10
	B = 0.1
)

var j = 20
var (
	k = "123"
	f = 0.1 // Hello
)

type S struct {
	a int `valid:"test"`
	b str
	c *T
}

func (s *S) m(a int) error {
	return nil
}

type I interface {
	m(a int) error
	// comment
	b(s str)
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
