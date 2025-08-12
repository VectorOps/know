package com.example;

import java.util.List;
import java.util.Map;

// Generic class
public class Box<T> {
    private T t;

    public void set(T t) {
        this.t = t;
    }

    public T get() {
        return t;
    }
}

// Generic interface
interface Pair<K, V> {
    K getKey();
    V getValue();
}

// Class implementing generic interface
class OrderedPair<K, V> implements Pair<K, V> {
    private K key;
    private V value;

    public OrderedPair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public K getKey() { return key; }
    public V getValue() { return value; }
}

// Generic method in a non-generic class
class Util {
    public static <K, V> boolean compare(Pair<K, V> p1, Pair<K, V> p2) {
        return p1.getKey().equals(p2.getKey()) &&
               p1.getValue().equals(p2.getValue());
    }
}

// Bounded type parameters
class BoundedBox<T extends Number> {
    private T t;

    public void set(T t) {
        this.t = t;
    }

    public T get() {
        return t;
    }
}

// Generic constructor
class MyGenericCtor {
    private double val;

    <T extends Number> MyGenericCtor(T t) {
        val = t.doubleValue();
    }
}
