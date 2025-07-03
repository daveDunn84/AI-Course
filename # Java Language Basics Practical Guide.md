# Java Language Basics Practical Guide

# Step 1: Understand the Goal

We want to:
Goal: The goal is to demonstrate Java basics through a zoo management simulation involving **caging**
animals using collections and generics, **feeding** them with functional interfaces and lambda expressions, and *
*logging** the process using custom annotations and reflection.
The simulation applies concepts like classes, objects, constructors, access modifiers, interfaces, streams, exception
handling, and method references.

# Step 2: Create the Folder Structure

``` 
ZooManagementExample/
└── src/
└── JavaLanguageBasics/
├── Animal.java
├── Cage.java
├── FeedingStrategy.java (interface)
├── Feedable.java (interface)
├── Info.java (interface)
├── ZooLogger.java
└── ZooManagementExample.java
```


# Step 3: Build Animal Class

# Variable declaration in File 
_📄 In your Animal.java file:_

```java

public class Animal {

private String name;
private int age;
private boolean fed = false;

}

```

# Constructor

```java

public Animal(String name, int age) {
this.name = name;
this.age = age;
}

```

# Methods

```java

public void feed() {
    fed = true;
    System.out.println(name + " has been fed.");
}

public boolean isFed() {
    return fed;
}

public String getName() {
    return name;
}

```




# Step 4: Build Generic Cage class
_📄 In your Cage.java file:_
# Class declarartion

```java
public class Cage<T extends Animal> { 
}

```



# Fields (Instance Variables)

```java
private List<T> animals = new ArrayList<>();
```



# Methods

```java
public void addAnimal(T animal) {
animals.add(animal);
}

public List<T> getAnimals() {
return animals;
}
```


# Add the neccessary imports from java core library

```java
import java.util.ArrayList;
import java.util.List;

```

# Step 5: Build Interfaces

_📄 In your Feedable.java file:_

#  Interface and abstract method Declaration
```java
public interface Feedable {
void feed();
}
```


📄 In your FeedingStrategy.java file:
#  Functional Interface Declaration
```java
@FunctionalInterface
public interface FeedingStrategy {
void feed(Animal animal);
}
```


# Step 6 Implement Interface
📄_ In your Animal.java file:_

``` java
public class Animal implements Feedable {
```



# Step 7 Create Custom Annotation 
_In 📄 Info.java file:_

```java
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Info {
String author();
String date();
String description() default ""; // Now optional
}
```


# Add the neccessary imports from java core library

```java
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
```



# Step 8 Use Custom Annotation
_In 📄 ZooLogger.java file :_

```java
public class ZooLogger {
    @Info(author = "Petronella", date = "2025-06-10", description = "Main feeding logic")
    public void logFeeding() {
        System.out.println("Feeding in progress...");
    }
}
```




# Step 9 Lets bring it all together  
_In 📄 ZooManagementExample.java file:_

#  3. Main Class
The entry point of your program.
Public so it’s accessible by the JVM.
``` java
public class ZooManagementExample {
```


# 4. Main Method
throws Exception: because getMethod() can throw checked exceptions.
``` java
public static void main(String[] args) throws Exception {
```


# 5. Create Animal List

``` java
   List<Animal> animals = new ArrayList<>();
   animals.add(new Animal("Lion", 5));
   animals.add(new Animal("Elephant", 10));
   animals.add(new Animal("Zebra", 3));
```
# 6. Create a Cage (Generic Class)

 ``` java
   Cage<Animal> cage = new Cage<>();
   animals.forEach(animal -> cage.addAnimal(animal));
```
   

# 7. Call the Logger Method

``` java
new ZooLogger().logFeeding();
```


# 8. Functional Interface and Stream

``` java
FeedingStrategy strategy = animal -> animal.feed();
cage.getAnimals().stream()
.filter(animal -> !animal.isFed())
.forEach(animal -> strategy.feed(animal));
```


# 10. Using Reflection to Read Annotation
``` java
 Method logMethod = ZooLogger.class.getMethod("logFeeding");
   if (logMethod.isAnnotationPresent(Info.class)) {
   Info info = logMethod.getAnnotation(Info.class);
   System.out.println("Author: " + info.author() + ", Date: " + info.date()
   + ", Description: " + info.description());
   }
```
  



# Add the neccessary imports from java core library

```java
import java.lang.reflect.Method;
import java.util.*;
```

