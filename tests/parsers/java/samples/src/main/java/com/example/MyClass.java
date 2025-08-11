package com.example;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import com.example.util.AnotherClass;

/**
 * A custom annotation.
 */
@interface MyAnnotation {
    String value();
}

/**
 * This is a Javadoc for MyClass.
 */
@MyAnnotation("class-level")
public class MyClass implements MyInterface {
    private static final String GREETING = "Hello";
    @Deprecated
    protected int count;
    private AnotherClass ac;

    /**
     * Javadoc for constructor.
     */
    public MyClass(int initialCount) {
        this.count = initialCount;
        this.ac = new AnotherClass();
    }

    /**
     * A simple method.
     * @param name The name to greet.
     * @return A greeting string.
     */
    public String greet(String name) throws java.io.IOException {
        return GREETING + ", " + name;
    }

    @Override
    public void doSomething() {
        // empty
    }
}

/**
 * A test enum.
 */
public enum Planet {
    MERCURY,
    VENUS,
    EARTH,
    MARS;

    /**
     * Javadoc for field.
     */
    private int mass;

    /**
     * Javadoc for enum constructor.
     */
    Planet() {
        this.mass = 1;
    }

    /**
     * Javadoc for enum method.
     */
    public int getMass() {
        return this.mass;
    }
}

/**
 * A test interface.
 */
interface MyInterface {
    int MY_CONSTANT = 42;

    /**
     * A method in the interface.
     */
    void doSomething();
}
