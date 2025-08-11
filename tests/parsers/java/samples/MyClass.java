package com.example;

import java.util.List;
import java.util.Map;

/**
 * This is a Javadoc for MyClass.
 */
public class MyClass {
    private static final String GREETING = "Hello";
    protected int count;

    /**
     * Javadoc for constructor.
     */
    public MyClass(int initialCount) {
        this.count = initialCount;
    }

    /**
     * A simple method.
     * @param name The name to greet.
     * @return A greeting string.
     */
    public String greet(String name) {
        return GREETING + ", " + name;
    }
}
