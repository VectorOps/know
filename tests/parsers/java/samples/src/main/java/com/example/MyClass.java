package com.example;

import java.util.List;
import java.util.Map;
import com.example.util.AnotherClass;

/**
 * This is a Javadoc for MyClass.
 */
public class MyClass {
    private static final String GREETING = "Hello";
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
}
