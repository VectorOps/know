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

// Simple type alias
type Foobar = int

type E struct {
	d str
}

type S struct {
	E
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

func SumIntsOrFloats[K comparable, V int64 | float64](m map[K]V) V {
	var s V
	for _, v := range m {
		s += v
	}
	return s
}

type Number interface {
	int64 | float64
}

type G struct {
	fmt.E
}
