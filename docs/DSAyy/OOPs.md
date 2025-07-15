
## Index

1. [OOP Fundamentals & Pillars](#1-oop-fundamentals--pillars)
    
2. [Classes & Objects](#2-classes--objects)
    
3. [Encapsulation & Access Control](#3-encapsulation--access-control)
    
4. [Inheritance](#4-inheritance)
    
5. [Polymorphism](#5-polymorphism)
    
6. [Abstraction & Interfaces](#6-abstraction--interfaces)
    
7. [Constructors, Destructors & Object Lifecycle](#7-constructors-destructors--object-lifecycle)
    
8. [SOLID Design Principles](#8-solid-design-principles)
    
9. [Key Design Patterns](#9-key-design-patterns)
    
10. [UML & System‑Level OOP Design](#10-uml--system-level-oop-design)
    
11. [Common OOP Interview Questions](#11-common-oop-interview-questions)
    
12. [Tips for A+‑Level OOP Interviews](#12-tips-for-a-level-oop-interviews)
    

---

## 1. OOP Fundamentals & Pillars

**Object‑Oriented Programming** models software as interacting **objects** (data + behavior).

- **Encapsulation:** Bundling data (fields) and methods; hides internal state.
    
- **Abstraction:** Exposing only relevant interfaces; hides complexity.
    
- **Inheritance:** Reusing and extending behavior via parent–child relationships.
    
- **Polymorphism:** “Many forms” – same interface, different implementations.
    

**Common Interview Queries**

- “Explain the four pillars of OOP with examples.”
    
- “Why isn’t OOP just about classes and objects?”
    
- “How do these pillars interplay to improve maintainability?”
    

---

## 2. Classes & Objects

- **Class:** Blueprint defining fields and methods.
    
- **Object:** Runtime instance with its own state.
    

```java
class User {
    private String name;
    private int id;

    public User(int id, String name) {  // constructor
        this.id = id;
        this.name = name;
    }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
}
```

- **Static vs. Instance members:**
    
    - `static` fields/methods belong to the class, not instances.
        
    - Instance members vary per object.
        

**Pitfalls & Gotchas**

- Overusing `static` leads to global state and thread‑safety issues.
    
- Mutable shared objects can break encapsulation.
    

**Common Interview Queries**

- “What’s the difference between a class variable and an instance variable?”
    
- “How does the JVM lay out objects in memory?”
    

---

## 3. Encapsulation & Access Control

**Access Modifiers (Java/C#)**

- `private` – accessible only within class.
    
- `protected` – plus subclasses and same package.
    
- `public` – everywhere.
    
- _(package‑private)_ – default, visible in same package.
    

**Benefits**

- Prevents external code from putting object into invalid state.
    
- Allows internal refactoring without breaking clients.
    

```java
class Account {
    private double balance;

    public void deposit(double amt) {
        if (amt < 0) throw new IllegalArgumentException();
        balance += amt;
    }
    public double getBalance() { return balance; }
}
```

**Common Interview Queries**

- “Why use getters/setters instead of public fields?”
    
- “Explain tight vs. loose encapsulation.”
    

---

## 4. Inheritance

- **Single vs. Multiple Inheritance:**
    
    - Java/C# allow single‑class inheritance + multiple interfaces.
        
    - C++ supports multiple inheritance (danger: the Diamond Problem).
        

```java
class Animal {
    void eat() { … }
}
class Dog extends Animal {
    void bark() { … }
}
```

- **Method overriding:** Subclass redefines parent’s behavior.
    
- **`super` / `base`** to call parent implementation.
    

**Pitfalls**

- Over‑deep hierarchies: brittle and hard to change.
    
- Inherited implementation may not suit subclass; prefer composition.
    

**Common Interview Queries**

- “What’s the Liskov Substitution Principle?”
    
- “When is inheritance preferable to composition?”
    

---

## 5. Polymorphism

- **Compile‑time (Static) Polymorphism:**
    
    - Method overloading (same name, different signature).
        
- **Runtime (Dynamic) Polymorphism:**
    
    - Method overriding + virtual dispatch.
        

```java
class Shape {
    double area() { return 0; }
}
class Circle extends Shape {
    double radius;
    @Override
    double area() { return Math.PI * radius*radius; }
}
void printArea(Shape s) {
    System.out.println(s.area());  // calls appropriate override
}
```

**Pitfalls**

- Overloading vs. overriding confusion.
    
- Virtual calls have slight performance cost.
    

**Common Interview Queries**

- “How does Java implement dynamic dispatch under the hood?”
    
- “Give an example of static binding vs. dynamic binding.”
    

---

## 6. Abstraction & Interfaces

- **Abstract Class:** Can have both concrete and abstract methods; share code.
    
- **Interface:** Pure contract (Java 8+ allows `default` methods too).
    

```java
interface Payment {
    void pay(double amount);
}
class CreditCardPayment implements Payment {
    public void pay(double amount) { … }
}
```

**Why Use Interfaces?**

- Achieve polymorphism without inheritance.
    
- Allow multiple “types” (Java) since classes can implement many interfaces.
    

**Common Interview Queries**

- “When would you choose an abstract class over an interface?”
    
- “How do you simulate multiple inheritance in Java?”
    

---

## 7. Constructors, Destructors & Object Lifecycle

- **Constructor Overloading:** Multiple ways to initialize an object.
    
- **Copy Constructors (C++/Java clone):** Deep vs. shallow copy.
    
- **Destructors / Finalizers:**
    
    - C++: RAII – destructors free resources.
        
    - Java: `finalize()` discouraged; use try‑with‑resources or explicit close.
        

```cpp
class File {
  FILE* fp;
public:
  File(const char* name) { fp = fopen(name,"r"); }
  ~File() { if (fp) fclose(fp); }
};
```

**Common Interview Queries**

- “Explain RAII and how it avoids resource leaks.”
    
- “Why is Java’s `finalize()` problematic?”
    

---

## 8. SOLID Design Principles

1. **S**ingle Responsibility
    
2. **O**pen/Closed (open for extension, closed for modification)
    
3. **L**iskov Substitution (subtypes replace supertypes)
    
4. **I**nterface Segregation (many small interfaces)
    
5. **D**ependency Inversion (code to abstractions)
    

**Example (Open/Closed)**

- Add new behavior by implementing new class; don’t modify existing ones.
    

```java
interface Discount {
  double apply(double price);
}
class SeasonalDiscount implements Discount { … }
class Checkout {
  double total(Discount d, double price) {
    return d.apply(price);
  }
}
```

**Common Interview Queries**

- “Give an example violation of Liskov Substitution.”
    
- “How does Dependency Injection relate to the Dependency Inversion Principle?”
    

---

## 9. Key Design Patterns

|Pattern|Intent|Typical Use|
|---|---|---|
|Singleton|Single global instance|Logger, Configuration|
|Factory|Encapsulate object creation|Creating DB connections|
|Observer|Publish/subscribe for events|UI event handling, notification services|
|Strategy|Swap algorithms at runtime|Pluggable sorting, payment methods|
|Decorator|Add behavior dynamically|I/O streams, UI components|
|Adapter|Match incompatible interfaces|Integrating legacy code|
|Builder|Step‑by‑step construction of complex objects|Fluent APIs, constructing HTTP requests|

**Common Interview Queries**

- “Sketch the class diagram for Observer.”
    
- “When would you use a Factory vs. Abstract Factory?”
    
- “Implement Singleton in a thread‑safe way.”
    

---

## 10. UML & System‑Level OOP Design

- **Class Diagrams:** Show classes, fields, methods, relationships (inheritance, composition, aggregation).
    
- **Sequence Diagrams:** Object interactions over time.
    
- **Use Case & Activity Diagrams:** High‑level workflows.
    

**Sample Question:** Design the class diagram for an online food ordering flow:

- Entities: `User`, `Order`, `Restaurant`, `Menu`, `Payment`
    
- Relationships:
    
    - `User` 1—* `Order`
        
    - `Order` _—_ `MenuItem` (via `OrderItem`)
        
    - `Order` 1—1 `Payment`
        

Be prepared to discuss how you’d evolve the model (e.g., add `PromoCode`, support multiple payment methods via Strategy).

---

## 11. Common OOP Interview Questions

1. **Explain the difference between abstraction and encapsulation.**
    
2. **How do you prevent a class from being subclassed?** (Java: `final` class)
    
3. **What’s the difference between `equals()` and `==` in Java?**
    
4. **Describe how garbage collection works in your language.**
    
5. **How do you implement a thread‑safe Singleton?**
    
6. **What are mix‑ins or traits, and how are they used?**
    
7. **Explain interface versioning and evolving APIs without breaking clients.**
    

---

## 12. Tips for A+‑Level OOP Interviews

- **Think in abstractions:** Push most logic into interfaces/strategies, not monolithic classes.
    
- **Discuss trade‑offs:** Composition vs. inheritance, runtime vs. compile‑time polymorphism.
    
- **Draw diagrams:** Even rough UML on the whiteboard demonstrates clarity.
    
- **Show awareness of language specifics:** Memory model, GC vs. RAII, multithreading nuances.
    
- **Relate to scale:** How does your design handle thousands of simultaneous orders?
    

---

## 1. Explain the difference between **abstraction** and **encapsulation**

- **Encapsulation** is about **hiding implementation details** and protecting object state by exposing a controlled interface.
    
    - You make fields `private` and expose public getters/setters.
        
    - **Example**:
        
        ```java
        public class Account {
          private double balance;         // hidden
          public void deposit(double amt) {
            if (amt <= 0) throw new IllegalArgumentException();
            balance += amt;
          }
          public double getBalance() {    // controlled access
            return balance;
          }
        }
        ```
        
- **Abstraction** is about **modeling the essential aspects** of a concept while hiding irrelevant details.
    
    - You design an **interface** or **abstract class** that declares _what_ operations are possible, without specifying _how_ they’re done.
        
    - **Example**:
        
        ```java
        public interface PaymentGateway {
          Receipt charge(CardInfo card, double amount);
        }
        // caller doesn’t care if it’s Stripe, PayPal, or a mock
        ```
        

**Key distinction**:

- Encapsulation is a technique for enforcing abstraction boundaries.
    
- Abstraction is a design goal (“What does this object do?”); encapsulation is the mechanism (“How do I enforce that boundary?”).
    

---

## 2. How do you **prevent a class from being subclassed**?

- **Java**: mark the class `final`:
    
    ```java
    public final class ImmutablePoint {
      private final int x, y;
      public ImmutablePoint(int x, int y) { this.x = x; this.y = y; }
      public int getX() { return x; }
      public int getY() { return y; }
    }
    ```
    
- **C#**: use the `sealed` keyword:
    
    ```csharp
    public sealed class Logger {
      // cannot be inherited
    }
    ```
    
- **Why**:
    
    - Enforce invariants (e.g., immutability).
        
    - Prevent method overriding that could break base‑class contracts.
        

---

## 3. What’s the difference between `==` and `equals()` in Java?

|Operator/Method|Compares…|Example|
|---|---|---|
|`==`|**Reference identity** (are they the same object?)|`new String("x") == new String("x") // false`|
|`equals()`|**Logical equality** (if implemented)|`new String("x").equals(new String("x")) // true`|

- **Default `equals()`** (in `Object`) does the same as `==`.
    
- **Override `equals()`** (and `hashCode()`) for value‑based classes:
    
    ```java
    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Point)) return false;
      Point p = (Point)o;
      return x == p.x && y == p.y;
    }
    @Override
    public int hashCode() {
      return Objects.hash(x, y);
    }
    ```
    
- **Pitfall**: Failing to override `hashCode()` breaks hashed collections (e.g., `HashSet`, `HashMap`).
    

---

## 4. Describe how **garbage collection** works in Java

1. **Roots & Reachability**
    
    - JVM tracks “GC roots” (active threads, static fields, local vars).
        
    - Anything reachable from those roots is **live**; everything else is **garbage**.
        
2. **Generational GC**
    
    - **Young Generation**: new objects; collected frequently (minor GC).
        
    - **Old Generation**: long‑lived objects; collected less often (major GC).
        
3. **Algorithms**
    
    - **Mark‑Sweep**: Mark live objects, then sweep dead ones.
        
    - **Copying**: Move live objects to a new area, reclaim the rest.
        
    - **CMS / G1**: Concurrent collectors to reduce pause times.
        
4. **Tuning**
    
    - Choose GC algorithm based on throughput vs. latency needs (e.g., G1 for low‑pause).
        
    - Adjust heap sizes (`-Xms`, `-Xmx`) and pause thresholds.
        
5. **Pitfalls**
    
    - Over‑tuning can backfire—monitor real metrics (pause times, allocation rate).
        
    - Holding onto references (e.g., static caches) leads to memory leaks.
        

---

## 5. How do you implement a **thread‑safe Singleton**?

### 5.1 Classic Double‑Checked Locking

```java
public class Singleton {
  private static volatile Singleton instance;
  private Singleton() { }
  public static Singleton getInstance() {
    if (instance == null) {                   // 1st check (no lock)
      synchronized (Singleton.class) {
        if (instance == null) {               // 2nd check (with lock)
          instance = new Singleton();
        }
      }
    }
    return instance;
  }
}
```

- **`volatile`** ensures visibility of the initialized instance across threads.
    

### 5.2 Initialization‑on‑Demand Holder

```java
public class Singleton {
  private Singleton() { }
  private static class Holder {
    static final Singleton INSTANCE = new Singleton();
  }
  public static Singleton getInstance() {
    return Holder.INSTANCE;
  }
}
```

- JVM guarantees that the `Holder` class is loaded and initialized **lazily** and **thread‑safely** on first access.
    

---
## 6. Composition vs. Inheritance

**Q:** What’s the difference between composition and inheritance, and when would you choose one over the other?

### Answer

- **Inheritance** (“is‑a”):
    
    - A subclass **extends** a parent class and **inherits** its behavior/contract.
        
    - Tight coupling—changes in the base class ripple into all subclasses.
        
    - Example:
        
        ```java
        class Animal {
          void eat() { … }
        }
        class Dog extends Animal {         // Dog is‑a Animal
          void bark() { … }
        }
        ```
        
- **Composition** (“has‑a”):
    
    - An object **contains** instances of other classes to reuse behavior.
        
    - Looser coupling—internal implementation can change without affecting clients.
        
    - Example:
        
        ```java
        class Engine {
          void start() { … }
        }
        class Car {
          private final Engine engine;     // Car has‑a Engine
          Car(Engine e) { this.engine = e; }
          void drive() {
            engine.start();
            // …
          }
        }
        ```
        

### Trade‑Offs & When to Use

|Aspect|Inheritance|Composition|
|---|---|---|
|Coupling|Tight|Looser|
|Flexibility|Less (fixed hierarchy)|More (delegate at runtime)|
|Reuse|Inherited behavior|Wrapped behavior via interfaces|
|Change Impact|High (fragile base class)|Localized|

- **Use inheritance** only when there is a clear “is‑a” relationship and you want to share or override default behavior.
    
- **Prefer composition** for greater flexibility, to avoid deep hierarchies, and when you want to switch implementations at runtime (e.g., via interfaces).
    

---

## 7. Applying SOLID Principles in Code

**Q:** Pick one SOLID principle and show a “before vs. after” refactoring example.

### Answer: Open/Closed Principle

- **Before (violates O/C)**
    
    ```java
    class InvoiceCalculator {
      double calculate(Invoice inv, String type) {
        if (type.equals("STANDARD")) {
          return inv.getAmount();
        } else if (type.equals("DISCOUNTED")) {
          return inv.getAmount() * 0.9;
        }
        // add new types → modify this method
        return inv.getAmount();
      }
    }
    ```
    
- **After (adheres to O/C)**
    
    ```java
    interface InvoiceStrategy {
      double calculate(Invoice inv);
    }
    
    class StandardInvoice implements InvoiceStrategy {
      public double calculate(Invoice inv) {
        return inv.getAmount();
      }
    }
    
    class DiscountedInvoice implements InvoiceStrategy {
      public double calculate(Invoice inv) {
        return inv.getAmount() * 0.9;
      }
    }
    
    class InvoiceCalculator {
      private final InvoiceStrategy strategy;
      InvoiceCalculator(InvoiceStrategy s) { this.strategy = s; }
      double calculate(Invoice inv) {
        return strategy.calculate(inv);
      }
    }
    ```
    
- **Why This Is Better**
    
    - **Closed** for modification: you don’t change `InvoiceCalculator`.
        
    - **Open** for extension: add new `InvoiceStrategy` implementations without touching existing code.
        

---

## 8. Strategy Pattern in Practice

**Q:** How does the Strategy pattern work? Provide a code sample.

### Answer

- **Intent:** Define a family of interchangeable algorithms, encapsulate each, and make them interchangeable.
    

```java
// Strategy interface
interface SortingStrategy {
  void sort(int[] data);
}

// Concrete strategies
class QuickSort implements SortingStrategy {
  public void sort(int[] data) { /* quicksort impl */ }
}

class MergeSort implements SortingStrategy {
  public void sort(int[] data) { /* mergesort impl */ }
}

// Context
class Sorter {
  private final SortingStrategy strategy;
  Sorter(SortingStrategy strat) { this.strategy = strat; }
  void sort(int[] data) {
    strategy.sort(data);
  }
}

// Client usage
int[] arr = {5,3,8,1};
Sorter sorter = new Sorter(new QuickSort());
sorter.sort(arr); // uses QuickSort
```

- **When to Use:**
    
    - When you have multiple ways to perform an operation and want to switch at runtime (e.g., various payment processors, sorting algorithms, compression codecs).
        
    - Keeps the context class simple and decoupled from specific implementations.
        

---

## 9. Evolving Interfaces Without Breaking Clients

**Q:** You need to add a method to a widely‑used interface in Java—how do you avoid breaking all implementations?

### Answer

1. **Default Methods (Java 8+):**
    
    ```java
    public interface Notifier {
      void send(String msg);
      default void sendUrgent(String msg) {
        send("[URGENT] " + msg);
      }
    }
    ```
    
    - Existing implementors only needed to define `send()`, and automatically get `sendUrgent()`.
        
2. **Adapter / Wrapper:**
    
    - Provide an abstract **adapter** class with a no‑op implementation:
        
        ```java
        public abstract class NotifierAdapter implements Notifier {
          @Override public void send(String msg) { }
          // new methods get default stubs
          @Override public void sendUrgent(String msg) { }
        }
        ```
        
    - Clients extend `NotifierAdapter` instead of implementing `Notifier` directly.
        
3. **Versioned Interfaces:**
    
    - Create `NotifierV2` and maintain both for a transition period.
        

---

## 10. Mix‑ins / Traits & Multiple Inheritance Alternatives

**Q:** What are mix‑ins or traits, and how do you use them in languages that don’t support multiple inheritance?

### Answer

- **Mix‑in / Trait:** A reusable set of methods and/or fields that can be “mixed into” a class without forming a superclass.
    

1. **Java via Interfaces with Default Methods:**
    
    ```java
    interface AuditMixin {
      default void auditLog(String msg) {
        System.out.println("AUDIT: " + msg);
      }
    }
    class OrderService implements AuditMixin {
      void placeOrder() {
        auditLog("order placed");
      }
    }
    ```
    
2. **Python Multiple Inheritance:**
    
    ```python
    class AuditMixin:
        def audit_log(self, msg):
            print(f"AUDIT: {msg}")
    
    class OrderService(AuditMixin):
        def place_order(self):
            self.audit_log("order placed")
    ```
    
3. **C# via Extension Methods:**
    
    ```csharp
    public static class AuditExtensions {
      public static void AuditLog(this object obj, string msg) {
        Console.WriteLine($"AUDIT: {msg}");
      }
    }
    // Usage:
    // myService.AuditLog("something happened");
    ```
    

- **Why Use Mix‑ins/Traits:**
    
    - Add orthogonal behavior (logging, validation, auditing) to many classes without polluting your class hierarchy.
        
    - Avoid the diamond problem of true multiple inheritance.
        

## 11. Observer Pattern: Real‑Time Notifications

**Q:** How do you implement the Observer pattern to push real‑time order status updates to multiple subscribers?

### Answer

- **Intent:** Define a one‑to‑many dependency so that when one object changes state, all its dependents are notified automatically.
    

```java
// Observer interface
interface OrderObserver {
  void onOrderStatusChanged(long orderId, String status);
}

// Subject interface
interface OrderSubject {
  void registerObserver(OrderObserver o);
  void removeObserver(OrderObserver o);
  void notifyObservers();
}

// Concrete Subject
class Order implements OrderSubject {
  private final long id;
  private String status;
  private final List<OrderObserver> observers = new ArrayList<>();

  public Order(long id) { this.id = id; }
  public void setStatus(String status) {
    this.status = status;
    notifyObservers();
  }
  public String getStatus() { return status; }
  public long getId() { return id; }

  @Override
  public void registerObserver(OrderObserver o) {
    observers.add(o);
  }
  @Override
  public void removeObserver(OrderObserver o) {
    observers.remove(o);
  }
  @Override
  public void notifyObservers() {
    for (OrderObserver o : observers) {
      o.onOrderStatusChanged(id, status);
    }
  }
}

// Concrete Observer
class EmailNotifier implements OrderObserver {
  public void onOrderStatusChanged(long orderId, String status) {
    System.out.println("Email: Order " + orderId + " is now " + status);
  }
}

// Usage
Order order = new Order(123L);
order.registerObserver(new EmailNotifier());
order.setStatus("CONFIRMED");  // triggers notification
```

**Why It Matters for Zomato:**

- Enables loosely‑coupled updates (e.g., email, SMS, mobile push) whenever the order status changes.
    
- Supports dynamic addition/removal of channels without modifying the core `Order` class.
    

---

## 12. Dependency Injection & Inversion of Control

**Q:** Explain Dependency Injection (DI) and show how it improves testability.

### Answer

- **Dependency Injection:** You **inject** collaborators into a class instead of creating them internally.
    
- **Inversion of Control:** The class no longer controls instantiation; an external framework or factory does.
    

```java
// Service interface
interface PaymentProcessor {
  void process(double amount);
}

// Concrete implementation
class StripeProcessor implements PaymentProcessor {
  public void process(double amount) {
    System.out.println("Processing $" + amount + " via Stripe.");
  }
}

// Client class without DI (hard‑coded dependency)
class CheckoutBad {
  private final PaymentProcessor processor = new StripeProcessor(); // tight coupling
  public void checkout(double amount) {
    processor.process(amount);
  }
}

// Client class with DI (loose coupling)
class Checkout {
  private final PaymentProcessor processor;
  // Constructor injection
  public Checkout(PaymentProcessor processor) {
    this.processor = processor;
  }
  public void checkout(double amount) {
    processor.process(amount);
  }
}

// Wiring (could be done by Spring, Guice, etc.)
PaymentProcessor pp = new StripeProcessor();
Checkout chk = new Checkout(pp);
chk.checkout(49.99);
```

**Testability:**

- In tests, you inject a **mock** or **fake** `PaymentProcessor` to verify behavior without hitting external services.
    

```java
class FakeProcessor implements PaymentProcessor {
  boolean called = false;
  public void process(double amount) { called = true; }
}
// In your unit test:
FakeProcessor fp = new FakeProcessor();
Checkout chkTest = new Checkout(fp);
chkTest.checkout(10);
assertTrue(fp.called);
```

---

## 13. Prototype Pattern: Cloning Objects

**Q:** When and how would you use the Prototype pattern to create object copies?

### Answer

- **Intent:** Specify the kinds of objects to create using a prototypical instance, then clone it to produce new objects.
    

```java
// Prototype interface
interface CloneableOrder extends Cloneable {
  CloneableOrder clone();
}

// Concrete prototype
class Order implements CloneableOrder {
  private long id;
  private String customer;
  private List<String> items;

  public Order(long id, String cust, List<String> items) {
    this.id = id; this.customer = cust;
    this.items = new ArrayList<>(items);
  }

  @Override
  public Order clone() {
    try {
      Order copy = (Order) super.clone();
      copy.items = new ArrayList<>(this.items);  // deep copy
      return copy;
    } catch (CloneNotSupportedException e) {
      throw new AssertionError();
    }
  }
}

// Usage
Order prototype = new Order(0, "guest", List.of("Pizza", "Coke"));
Order newOrder = prototype.clone();
newOrder.setId(101L);
```

**When to Use:**

- Creating new orders with default templates (e.g., guest checkout).
    
- Performance benefit when object construction is expensive or complex.
    

---

## 14. Liskov Substitution Principle (LSP) Violation & Fix

**Q:** Show an LSP violation using a `Rectangle`/`Square` example, then refactor to comply.

### Answer

- **Violation:** `Square` inherits from `Rectangle` but cannot honor `setWidth`/`setHeight` independently.
    

```java
class Rectangle {
  protected int width, height;
  public void setWidth(int w) { width = w; }
  public void setHeight(int h) { height = h; }
  public int getArea() { return width * height; }
}

class Square extends Rectangle {
  @Override
  public void setWidth(int w) {
    super.setWidth(w); super.setHeight(w);
  }
  @Override
  public void setHeight(int h) {
    super.setWidth(h); super.setHeight(h);
  }
}

// Client code expecting Rectangle behavior
Rectangle r = new Square();
r.setWidth(5);
r.setHeight(10);
// Now r.getArea() == 100, not 50 → surprising behavior!
```

- **Fix:** Use composition instead of inheritance.
    

```java
interface Shape {
  int getArea();
}

class Rectangle implements Shape {
  private int width, height;
  /* setters/getters omitted */
  public int getArea() { return width * height; }
}

class Square implements Shape {
  private int side;
  public Square(int side) { this.side = side; }
  public int getArea() { return side * side; }
}
```

**Key Takeaway:**

- Subtypes must conform to the expectations set by their supertypes.
    
- Favor composition/interfaces when specialization changes the contract.
    

---

## 15. Chain of Responsibility: Validation Pipeline

**Q:** Design a request‑validation pipeline for orders using the Chain of Responsibility pattern.

### Answer

- **Intent:** Pass a request along a chain of handlers; each handler decides to process it and/or pass it on.
    

```java
// Handler interface
abstract class OrderHandler {
  protected OrderHandler next;
  public OrderHandler linkWith(OrderHandler next) {
    this.next = next; return next;
  }
  public abstract boolean handle(OrderContext ctx);
}

class StockCheckHandler extends OrderHandler {
  @Override
  public boolean handle(OrderContext ctx) {
    if (!checkStock(ctx)) return false;
    return next == null || next.handle(ctx);
  }
}

class PaymentValidationHandler extends OrderHandler {
  @Override
  public boolean handle(OrderContext ctx) {
    if (!validatePayment(ctx)) return false;
    return next == null || next.handle(ctx);
  }
}

class AddressValidationHandler extends OrderHandler {
  @Override
  public boolean handle(OrderContext ctx) {
    if (!validateAddress(ctx)) return false;
    return next == null || next.handle(ctx);
  }
}

// Wiring the chain
OrderHandler chain = new StockCheckHandler();
chain.linkWith(new PaymentValidationHandler())
     .linkWith(new AddressValidationHandler());

// Processing
boolean valid = chain.handle(orderCtx);
if (!valid) {
  // reject order
}
```

**Why It Fits Zomato:**

- You can add/remove/ reorder validation steps (stock, payment, address, fraud check) without changing the core order‑processing logic.
    

------

## 16. Interface Segregation Principle (ISP) Violation & Refactor

**Q:** You have this bloated interface—`MultiFunctionDevice`—that forces clients to implement methods they don’t need. How do you refactor it to obey ISP?

### Violation

```java
public interface MultiFunctionDevice {
  void print(Document d);
  void fax(Document d);
  void scan(Document d);
}

class OldPrinter implements MultiFunctionDevice {
  public void print(Document d) { /* okay */ }
  public void fax(Document d)   { throw new UnsupportedOperationException(); }
  public void scan(Document d)  { throw new UnsupportedOperationException(); }
}
```

_Problem_: `OldPrinter` is forced to implement fax/scan it doesn’t support.

### Refactor

```java
// Split into focused interfaces
public interface Printer {
  void print(Document d);
}
public interface Scanner {
  void scan(Document d);
}
public interface Fax {
  void fax(Document d);
}

// Now compose only what you need
class SimplePrinter implements Printer {
  public void print(Document d) { /* print logic */ }
}
class MultiFunctionPrinter implements Printer, Scanner, Fax {
  public void print(Document d) { /* ... */ }
  public void scan(Document d)  { /* ... */ }
  public void fax(Document d)   { /* ... */ }
}
```

**Why This Matters**

- Clients depend **only** on the methods they use.
    
- Future devices can implement just `Printer` or `Scanner` without stub methods.
    

---

## 17. Dependency Inversion Principle (DIP) in Practice

**Q:** Show how to apply DIP so a high‑level module doesn’t depend on a low‑level implementation.

### Before (Violation)

```java
class EmailService {
  private final SmtpClient smtp = new SmtpClient("smtp.zomato.com");
  void sendEmail(String to, String body) {
    smtp.connect();
    smtp.send(to, body);
  }
}
```

_Problem_: `EmailService` is tightly coupled to `SmtpClient`.

### After (Adheres to DIP)

```java
// Abstraction
public interface MessageSender {
  void send(String to, String body);
}

// Low‑level module
public class SmtpSender implements MessageSender {
  public void send(String to, String body) {
    // SMTP logic
  }
}

// High‑level module depends on abstraction
public class NotificationService {
  private final MessageSender sender;
  public NotificationService(MessageSender sender) {
    this.sender = sender;
  }
  public void notifyUser(String user, String msg) {
    sender.send(user, msg);
  }
}

// Wiring (e.g., in a DI container)
MessageSender smtp = new SmtpSender();
NotificationService svc = new NotificationService(smtp);
```

**Why This Matters**

- Swapping to an SMS or push‑notification provider requires **no changes** in `NotificationService`.
    
- Improves **testability** by injecting mocks for `MessageSender`.
    

---

## 18. Generics & Wildcards in Java (“PECS”)

**Q:** Explain covariance/contravariance in Java generics and the PECS mnemonic.

### Key Concepts

- **Producer Extends** (`? extends T`): You can **read** `T` out safely, but cannot write.
    
- **Consumer Super** (`? super T`): You can **write** `T` into it, but reading gives `Object`.
    

```java
List<Integer> ints = List.of(1,2,3);
List<? extends Number> producers = ints;
// producers.add(4);              // ❌ compile‑error
Number n = producers.get(0);       // ✅ safe

List<Object> objs = new ArrayList<>();
List<? super Integer> consumers = objs;
consumers.add(42);                 // ✅ safe
Object o = consumers.get(0);       // ✅ gives Object
Integer i = consumers.get(0);      // ❌ need cast
```

### Use Cases

- **`? extends`** for read‑only inputs (e.g., passing a `List<Integer>` to a method expecting numbers).
    
- **`? super`** for write‑only outputs (e.g., collecting elements).
    

**Why This Matters**

- Correct use of wildcards avoids unsafe casts and maximizes API flexibility.
    

---

## 19. Decorator Pattern for Menu Item Pricing

**Q:** You need to add optional toppings (cheese, olives, etc.) to a base pizza and calculate the total price dynamically. How do you use the Decorator pattern?

### Implementation

```java
// Component
interface MenuItem {
  String getDescription();
  double getPrice();
}

// Concrete Component
class PlainPizza implements MenuItem {
  public String getDescription() { return "Plain Pizza"; }
  public double getPrice() { return 5.00; }
}

// Base Decorator
abstract class ToppingDecorator implements MenuItem {
  protected final MenuItem item;
  public ToppingDecorator(MenuItem item) { this.item = item; }
}

// Concrete Decorators
class Cheese extends ToppingDecorator {
  public Cheese(MenuItem item) { super(item); }
  public String getDescription() {
    return item.getDescription() + ", Cheese";
  }
  public double getPrice() {
    return item.getPrice() + 1.25;
  }
}

class Olives extends ToppingDecorator {
  public Olives(MenuItem item) { super(item); }
  public String getDescription() {
    return item.getDescription() + ", Olives";
  }
  public double getPrice() {
    return item.getPrice() + 0.75;
  }
}

// Usage
MenuItem order = new Olives(new Cheese(new PlainPizza()));
System.out.println(order.getDescription()); // Plain Pizza, Cheese, Olives
System.out.println(order.getPrice());       // 7.00
```

**Why This Matters**

- You can stack **any combination** of toppings at runtime without exploding the class count.
    
- Promotes the **open/closed principle**—new toppings are new decorators, no changes to existing code.
    

---

## 20. Adapter Pattern for Legacy Payment Integration

**Q:** You have an existing `PaymentGateway` interface but need to integrate a legacy processor with an incompatible API. How do you adapt it?

### Existing Interface

```java
interface PaymentGateway {
  Receipt pay(CreditCard card, double amount);
}
```

### Legacy Class

```java
class LegacyProcessor {
  public TransactionResult processPayment(String cardNum, String exp, double amt) {
    // old style logic…
  }
}
```

### Adapter

```java
class LegacyPaymentAdapter implements PaymentGateway {
  private final LegacyProcessor legacy;
  public LegacyPaymentAdapter(LegacyProcessor legacy) {
    this.legacy = legacy;
  }
  @Override
  public Receipt pay(CreditCard card, double amount) {
    TransactionResult res = legacy.processPayment(
      card.getNumber(),
      card.getExpiry(),
      amount
    );
    return new Receipt(res.getId(), res.isSuccessful());
  }
}

// Usage
PaymentGateway gateway = new LegacyPaymentAdapter(new LegacyProcessor());
Receipt rcpt = gateway.pay(card, 100.00);
```

**Why This Matters**

- The adapter shields the rest of your codebase from the legacy API.
    
- You can replace the adapter later when you migrate to a modern processor without touching business logic.
    

---

## 21. Builder Pattern for Complex Object Construction

**Q:** You need to construct an `Order` object that has dozens of optional parameters (promo code, special instructions, tip, delivery window, etc.). How do you apply the Builder pattern to keep construction readable and safe?

### Answer

- **Intent:** Separate the construction of a complex object from its representation, allowing step‑by‑step creation and immutability.
    

```java
public class Order {
  private final long    orderId;
  private final long    userId;
  private final List<Item> items;
  private final String  promoCode;
  private final String  instructions;
  private final double  tip;
  private final LocalDateTime deliverBy;

  private Order(Builder b) {
    this.orderId      = b.orderId;
    this.userId       = b.userId;
    this.items        = List.copyOf(b.items);
    this.promoCode    = b.promoCode;
    this.instructions = b.instructions;
    this.tip          = b.tip;
    this.deliverBy    = b.deliverBy;
  }

  // Getters omitted for brevity…

  public static class Builder {
    // Required
    private final long    orderId;
    private final long    userId;
    private final List<Item> items;

    // Optional – initialized to defaults
    private String          promoCode    = "";
    private String          instructions = "";
    private double          tip          = 0.0;
    private LocalDateTime   deliverBy    = LocalDateTime.now().plusHours(1);

    public Builder(long orderId, long userId, List<Item> items) {
      this.orderId = orderId;
      this.userId  = userId;
      this.items   = new ArrayList<>(items);
    }

    public Builder promoCode(String code) {
      this.promoCode = code; return this;
    }
    public Builder instructions(String instr) {
      this.instructions = instr; return this;
    }
    public Builder tip(double t) {
      this.tip = t; return this;
    }
    public Builder deliverBy(LocalDateTime dt) {
      this.deliverBy = dt; return this;
    }

    public Order build() {
      // Validate invariants if needed
      if (items.isEmpty()) {
        throw new IllegalStateException("Order must have at least one item");
      }
      return new Order(this);
    }
  }
}

// Usage:
Order order = new Order.Builder(1001, 2002, itemsList)
    .promoCode("ZOM10")
    .tip(2.50)
    .instructions("Leave at door")
    .build();
```

**Why It Matters:**

- Keeps constructors concise even with many parameters.
    
- Ensures immutability and validation before object creation.
    
- Fluent API makes client code readable and maintainable.
    

---

## 22. Template Method Pattern for Order Processing

**Q:** You have a generic algorithm for processing orders (validate → charge → notify) but want to allow subclasses to customize individual steps. How does Template Method help?

### Answer

- **Intent:** Define the skeleton of an algorithm in a base class, deferring specific steps to subclasses.
    

```java
// Abstract template
public abstract class OrderProcessor {
  // Template method
  public final void process(Order o) {
    validate(o);
    charge(o);
    notifyCustomer(o);
  }

  protected abstract void validate(Order o);

  protected void charge(Order o) {
    // default charge logic, e.g., call payment gateway
    System.out.println("Charging order " + o.getId());
  }

  protected abstract void notifyCustomer(Order o);
}

// Concrete subclass
public class DigitalOrderProcessor extends OrderProcessor {
  @Override
  protected void validate(Order o) {
    // e.g., check digital SKU stock
  }

  @Override
  protected void notifyCustomer(Order o) {
    // e.g., send email with download link
  }
}
```

**Why It Matters:**

- Enforces overall workflow while permitting customization at well‑defined “hooks.”
    
- Prevents duplication of common steps across multiple processors.
    

---

## 23. Proxy Pattern for Lazy‑Loading Customer Profile

**Q:** Loading a `Customer` object eagerly pulls in a large `Profile` from a remote service. How can you use Proxy to defer that call until actually needed?

### Answer

- **Intent:** Provide a surrogate or placeholder for another object to control access, e.g., lazy initialization.
    

```java
// Subject interface
public interface Customer {
  String getName();
  Profile getProfile();
}

// Real subject
class RealCustomer implements Customer {
  private final Profile profile;
  // constructor loads the profile immediately…
  public RealCustomer(int id) {
    this.profile = remoteService.fetchProfile(id);
  }
  public String getName() { /* … */ }
  public Profile getProfile() { return profile; }
}

// Proxy
class CustomerProxy implements Customer {
  private final int id;
  private Profile profile;           // initially null

  public CustomerProxy(int id) { this.id = id; }
  public String getName() {
    // name is cached or cheap; load if necessary
    return remoteService.fetchName(id);
  }
  public Profile getProfile() {
    if (profile == null) {
      profile = remoteService.fetchProfile(id);  // lazy load
    }
    return profile;
  }
}

// Usage
Customer cust = new CustomerProxy(42);
System.out.println(cust.getName());          // no profile call
Profile p = cust.getProfile();               // remote fetch happens here
```

**Why It Matters:**

- Reduces remote calls and memory when profile data is seldom used.
    
- Transparently swaps in the proxy without changing client code.
    

---

## 24. Flyweight Pattern for Menu Items

**Q:** Zomato caches menu items for millions of restaurants, yet many share identical metadata (name, description). How do you use Flyweight to reduce memory?

### Answer

- **Intent:** Share as much intrinsic data as possible among similar objects, storing only extrinsic state externally.
    

```java
// Flyweight
class MenuItemType {
  private final String name;
  private final String description;
  // shared across many restaurants
  public MenuItemType(String name, String desc) {
    this.name = name; this.description = desc;
  }
  // getters…
}

// Factory to manage shared types
class MenuItemFactory {
  private static final Map<String, MenuItemType> cache = new ConcurrentHashMap<>();

  public static MenuItemType getType(String name, String desc) {
    return cache.computeIfAbsent(name, k -> new MenuItemType(name, desc));
  }
}

// Context objects
class MenuItem {
  private final MenuItemType type;  // intrinsic (shared)
  private final double price;       // extrinsic (varies by restaurant)
  public MenuItem(String name, String desc, double price) {
    this.type  = MenuItemFactory.getType(name, desc);
    this.price = price;
  }
  // getters delegate to type…
}
```

**Why It Matters:**

- Intrinsic data (name, description) stored once per unique item, not per restaurant.
    
- Dramatically cuts memory for large, global catalogs.
    

---

## 25. Visitor Pattern for Diverse Operations

**Q:** You have a hierarchy (`Pizza`, `Burger`, `Sushi`) and need to perform operations like pricing, nutrition calculation, and serialization without cluttering classes. How does Visitor help?

### Answer

- **Intent:** Encapsulate operations on a set of object structures by moving them into visitor classes.
    

```java
// Element interface
interface MenuElement {
  <R> R accept(MenuVisitor<R> visitor);
}

class Pizza implements MenuElement {
  // fields…
  public <R> R accept(MenuVisitor<R> v) {
    return v.visit(this);
  }
}

class Burger implements MenuElement {
  public <R> R accept(MenuVisitor<R> v) {
    return v.visit(this);
  }
}

// Visitor interface
interface MenuVisitor<R> {
  R visit(Pizza p);
  R visit(Burger b);
  // ... other menu types
}

// Concrete visitor: price calculator
class PriceCalculator implements MenuVisitor<Double> {
  public Double visit(Pizza p) {
    return basePrice(p) + toppingCost(p);
  }
  public Double visit(Burger b) {
    return basePrice(b) + extrasCost(b);
  }
}

// Usage
List<MenuElement> menu = List.of(new Pizza(...), new Burger(...));
double total = menu.stream()
    .mapToDouble(e -> e.accept(new PriceCalculator()))
    .sum();
```

**Why It Matters:**

- Adds new operations without modifying element classes.
    
- Separates algorithms from data structures, supporting open/closed.
    

---
