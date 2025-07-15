## Index

1. [Introduction to DBMS](#1-introduction-to-dbms)
    
2. [Database Architecture](#2-database-architecture)
    
3. [Keys and Constraints](#3-keys-and-constraints)
    
4. [SQL vs. NoSQL](#4-sql-vs-nosql)
    
5. [Normalization and Denormalization](#5-normalization-and-denormalization)
    
6. [Indexing](#6-indexing)
    
7. [Transactions & ACID Properties](#7-transactions--acid-properties)
    
8. [Concurrency Control & Isolation Levels](#8-concurrency-control--isolation-levels)
    
9. [Deadlocks & Resolution Strategies](#9-deadlocks--resolution-strategies)
    
10. [Joins & Set Operations](#10-joins--set-operations)
    
11. [Stored Procedures, Functions & Triggers](#11-stored-procedures-functions--triggers)
    
12. [Views & Materialized Views](#12-views--materialized-views)
    
13. [Database Design & ER Diagrams](#13-database-design--er-diagrams)
    
14. [Commonly Asked Interview Questions](#14-commonly-asked-interview-questions)
    
15. [Tips for Zomato Interview](#15-tips-for-zomato-interview)
    

---

## 1. Introduction to DBMS

- **Definition**: A Database Management System (DBMS) is software that enables creation, management, and querying of databases.
    
- **Purpose**: Provides an abstraction layer over raw data files, ensuring data integrity, security, and efficient access.
    
- **Examples**: MySQL, PostgreSQL, Oracle, SQL Server for relational; MongoDB, Cassandra for NoSQL.
    

---

## 2. Database Architecture

- **Three-Level ANSI/SPARC Model**
    
    1. **Internal Level**: Physical storage (pages, blocks on disk).
        
    2. **Conceptual Level**: Logical schema (tables, relationships).
        
    3. **External Level**: User views (restricted subsets via views).
        
- **Client–Server**: Clients send SQL queries; server processes and returns results.
    
- **Tiered Architectures**:
    
    - **Two-tier**: Client ↔ DB server.
        
    - **Three-tier**: Client ↔ App server ↔ DB server.
        

---

## 3. Keys and Constraints

- **Primary Key**: Unique identifier for table rows; non-null.
    
- **Candidate Key**: Any minimal set of columns that can uniquely identify rows.
    
- **Foreign Key**: Enforces referential integrity between tables.
    
- **Unique Constraint**: Ensures all values in a column (or group) are unique.
    
- **Check Constraint**: Validates data against a Boolean expression.
    

---

## 4. SQL vs. NoSQL

|Aspect|SQL|NoSQL|
|---|---|---|
|Schema|Fixed, predefined|Dynamic, flexible|
|Data Model|Tables (rows & columns)|Document, Key-Value, Column, Graph|
|Transactions|Strong ACID support|BASE (Basic Availability, Soft state, Eventually consistent)|
|Scaling|Vertical (scale-up)|Horizontal (scale-out)|
|Use Cases|Complex queries, transactions|High throughput, flexible schema|

---

## 5. Normalization and Denormalization

- **Normalization**: Process to eliminate redundancy & anomalies via Normal Forms (1NF, 2NF, 3NF, BCNF).
    
    - _1NF_: Atomic values.
        
    - _2NF_: No partial dependencies on a candidate key.
        
    - _3NF_: No transitive dependencies.
        
- **Denormalization**: Intentionally introducing redundancy for performance gains (e.g., read-heavy workloads).
    

---

## 6. Indexing

- **Purpose**: Speed up data retrieval by creating auxiliary data structures.
    
- **Types**:
    
    - **Clustered Index**: Alters physical order of data. Only one per table.
        
    - **Non-Clustered Index**: Separate structure with pointers to rows. Multiple per table.
        
    - **Hash Index**: Uses hash table for exact-match lookups.
        
    - **Bitmap Index**: Efficient for low-cardinality columns (e.g., gender).
        
- **Trade-offs**:
    
    - **Pros**: Fast reads.
        
    - **Cons**: Slower writes/updates; additional storage.
        
- **Interview Tip**: Explain when to add composite indexes vs. single-column indexes, and how covering indexes reduce lookups.
    

---

## 7. Transactions & ACID Properties

- **Transaction**: A unit of work that is either fully complete or fully rolled back.
    
- **ACID**:
    
    1. **Atomicity**: All-or-nothing execution.
        
    2. **Consistency**: Database remains in a valid state.
        
    3. **Isolation**: Concurrent transactions don’t interfere.
        
    4. **Durability**: Once committed, results persist.
        
- **Commit vs. Rollback**: Persist or undo transaction changes.
    

---

## 8. Concurrency Control & Isolation Levels

- **Locking**:
    
    - **Shared Lock (S)**: Read-only.
        
    - **Exclusive Lock (X)**: Read/write.
        
- **Multi-Version Concurrency Control (MVCC)**: Maintains multiple versions for readers.
    
- **Isolation Levels** (SQL Standard):
    
    1. **Read Uncommitted**: Dirty reads allowed.
        
    2. **Read Committed**: No dirty reads.
        
    3. **Repeatable Read**: No non-repeatable reads.
        
    4. **Serializable**: Full isolation.
        
- **Anomalies**: Dirty reads, non-repeatable reads, phantom reads.
    

---

## 9. Deadlocks & Resolution Strategies

- **Deadlock**: Two or more transactions waiting indefinitely for each other’s locks.
    
- **Detection**: Wait-for graph cycle detection.
    
- **Prevention**: Lock ordering, timeouts, or no-wait policies.
    
- **Resolution**: Automatically roll back one transaction (victim).
    

---

## 10. Joins & Set Operations

- **Joins**:
    
    - **Inner Join**: Intersection of tables.
        
    - **Left/Right Outer Join**: All from one side + matching from the other.
        
    - **Full Outer Join**: All rows, matching where possible.
        
    - **Cross Join**: Cartesian product.
        
- **Set Operations**:
    
    - **UNION / UNION ALL**: Combine result sets (UNION removes duplicates).
        
    - **INTERSECT**: Common rows.
        
    - **EXCEPT/MINUS**: Rows in first not in second.
        

---

## 11. Stored Procedures, Functions & Triggers

- **Stored Procedure**: Precompiled group of SQL statements; can perform complex operations.
    
- **Function**: Returns a value; can be used in SQL expressions.
    
- **Trigger**: Auto-executes upon DML events (INSERT/UPDATE/DELETE).
    
- **Use Case**: Encapsulate business logic, enforce data integrity.
    

---

## 12. Views & Materialized Views

- **View**: Virtual table defined by a query; no storage overhead.
    
- **Materialized View**: Stores query result physically; must be refreshed.
    
- **Benefits**: Simplify complex queries, enforce security, improve performance (materialized).
    

---

## 13. Database Design & ER Diagrams

- **ER Modeling**: Entities (tables), Attributes (columns), Relationships (1:1, 1:N, N:N).
    
- **Design Steps**:
    
    1. Requirement gathering.
        
    2. Conceptual model (ER diagram).
        
    3. Logical model (relational schema).
        
    4. Physical model (indexes, storage).
        
- **Zomato Context**: Model users, restaurants, orders, menus, ratings with clear relationships.
    

---

## 14. Commonly Asked Interview Questions

1. **Difference between DELETE and TRUNCATE**
    
    - DELETE logs individual row deletions & can be rolled back; TRUNCATE deallocates pages, faster, cannot be rolled back in most DBs.
        
2. **Explain ACID vs. BASE**
    
    - ACID (strict consistency), BASE (eventual consistency suited for high-scale NoSQL).
        
3. **When would you choose NoSQL over SQL?**
    
    - Flexible schema, massive horizontal scalability, simple key-value access.
        
4. **What is a covering index?**
    
    - An index that contains all columns needed by a query, avoiding lookup to base table.
        
5. **How do you detect and handle deadlocks?**
    
    - Monitor wait-for graphs; set lock timeouts; implement retry logic.
        
6. **Explain isolation levels with examples of anomalies.**
    
7. **Describe normalization up to 3NF with examples.**
    
8. **How do you optimize slow SQL queries?**
    
    - Analyze execution plan, add appropriate indexes, rewrite joins/subqueries, use pagination.
        
9. **What is MVCC and how does it work?**
    
10. **Design a table schema for storing user ratings and reviews.**
    

---

## 15. Tips for Zomato Interview

- **Relate Answers to Scale**: Emphasize how indexing, partitioning, and sharding help handle Zomato’s high-traffic order and search workloads.
    
- **Use Real-World Metrics**: “At peak hours, we process X QPS; an index can reduce I/O by Y%.”
    
- **Discuss Data Consistency**: Balance between strong consistency for payments vs. eventual consistency for analytics.
    
- **Be Ready to Whiteboard**: Sketch ER diagrams quickly for restaurants–menus–orders.
    
- **Behavioral Tie-In**: Show how you’ve debugged slow queries or optimized a schema in past projects.
    

---
# More Detailed

Below are the first five topics expanded with deeper explanations, practical examples, and “common interview queries” you might encounter.

---

## 1. Introduction to DBMS

### What is a DBMS?

A **Database Management System** is software that:

- **Stores** large volumes of structured data efficiently on disk.
    
- **Manages** concurrent access by multiple users or applications.
    
- **Enforces** data integrity, security, and recovery from crashes.
    

### Core Components

1. **Storage Manager**
    
    - Handles data files, buffer management, and I/O.
        
2. **Query Processor**
    
    - Parses SQL, optimizes it into an execution plan, and runs it.
        
3. **Transaction Manager**
    
    - Ensures ACID properties (covered later).
        
4. **Metadata Catalog**
    
    - Keeps schema definitions, statistics, and user privileges.
        

### Why Use a DBMS?

- **Abstraction**: Hides file‐level details (you write SQL, not file reads).
    
- **Safety**: Built‐in backup/restore and crash recovery.
    
- **Concurrency**: Multiple clients can read/write safely.
    
- **Security**: Fine‐grained user/role privileges on tables, rows, columns.
    

### Common Interview Queries

- **“List DBMS advantages over flat files.”**
    
- **“Explain the role of the buffer manager.”**
    
- **“What’s in the system catalog?”**
    
- **SQL exercise**:
    
    ```sql
    -- How to list all tables and their row counts?
    SELECT table_schema, table_name,
           table_rows 
      FROM information_schema.tables
     WHERE table_type='BASE TABLE';
    ```
    

---

## 2. Database Architecture

### ANSI/SPARC Three‑Level Model

- **Internal Level**
    
    - Physical storage details (page layouts, indexing structures).
        
- **Conceptual Level**
    
    - Global logical schema: tables, columns, data types, relationships.
        
- **External Level**
    
    - Multiple user “views” (e.g., read‐only or reduced‐column subsets).
        

### Client‑Server & N‑Tier

- **Two‑Tier**: Client directly issues SQL to DB server. Simpler but less scalable.
    
- **Three‑Tier**: Client → Application server → Database. Better separation of concerns, caching, and load‑balancing.
    

### Distributed & Cloud Architectures

- **Sharding**: Horizontal partitioning by key (e.g., user_id ranges).
    
- **Replication**: Master–slave or multi‑master copies for high availability.
    
- **Cloud DB Services**: Aurora (MySQL‑compatible), DynamoDB (NoSQL), BigQuery (analytical).
    

### Common Interview Queries

- **“Draw and explain the ANSI/SPARC model.”**
    
- **“When would you choose two‑tier vs. three‑tier?”**
    
- **“How does sharding differ from replication?”**
    
- **SQL exercise**:
    
    ```sql
    -- Check replication status in MySQL
    SHOW SLAVE STATUS\G;
    ```
    

---

## 3. Keys and Constraints

### Types of Keys

- **Primary Key**
    
    - Uniquely identifies each row; cannot be NULL.
        
- **Candidate Key**
    
    - Any minimal superkey; one is chosen as the PK.
        
- **Foreign Key**
    
    - References PK in another table; enforces referential integrity.
        
- **Composite Key**
    
    - PK spanning multiple columns (e.g., `(order_id, item_id)`).
        

### Constraint Types

- **UNIQUE**: No duplicates allowed.
    
- **NOT NULL**: Disallows NULLs.
    
- **CHECK**: Enforces a Boolean expression (e.g., `CHECK (age>=18)`).
    
- **DEFAULT**: Supplies default value if none given.
    

### How Constraints Are Enforced

- At **INSERT** or **UPDATE**, the DBMS checks each relevant constraint.
    
- Violations cause the entire statement to **roll back**.
    

### Common Interview Queries

- **“Write SQL to add a foreign key to an existing table.”**
    
- **“How do composite keys affect indexing?”**
    
- **“Explain deferred vs. immediate constraint checking.”**
    
- **SQL exercises**:
    
    ```sql
    -- Add a FK:
    ALTER TABLE orders
      ADD CONSTRAINT fk_user
      FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE;
    
    -- Create a table with composite PK:
    CREATE TABLE order_items (
      order_id INT,
      item_id  INT,
      qty      INT,
      PRIMARY KEY (order_id, item_id)
    );
    ```
    

---

## 4. SQL vs. NoSQL

|Feature|SQL (Relational)|NoSQL (Non‑relational)|
|---|---|---|
|**Schema**|Rigid: ALTER TABLE required|Flexible: Documents/columns added freely|
|**Data Model**|Rows & columns|Document (JSON), Key‑Value, Graph, Column|
|**Transactions**|Full ACID|BASE: eventual consistency|
|**Scaling**|Vertical (bigger machines)|Horizontal (add commodity nodes)|
|**Use Cases**|Financial, inventory systems|High‑scale web services, caching|

### Key Trade‑offs

- **Joins & Complex Queries**: SQL excels.
    
- **Rapid Schema Evolution**: NoSQL shines.
    
- **Strong Consistency**: SQL by default.
    
- **Massive Scale‑out**: Many NoSQL engines optimize for this.
    

### Common Interview Queries

- **“When would you choose a document store over a relational DB?”**
    
- **“Explain BASE: break down each letter.”**
    
- **“How does eventual consistency manifest in reads?”**
    
- **Hands‑on**:
    
    ```sql
    -- Relational JOIN example:
    SELECT u.name, o.total
      FROM users u
      JOIN orders o ON u.id = o.user_id
     WHERE o.created_at > '2025-01-01';
    
    -- MongoDB equivalent:
    db.orders.aggregate([
      { $match: { created_at: { $gt: ISODate("2025-01-01") } } },
      { $lookup: {
          from: "users",
          localField: "user_id",
          foreignField: "_id",
          as: "user"
      }},
      { $unwind: "$user" },
      { $project: { "user.name":1, total:1 } }
    ]);
    ```
    

---

## 5. Normalization and Denormalization

### Goals of Normalization

- **Eliminate Redundancy**: Avoid repeating data.
    
- **Prevent Anomalies**: INSERT, UPDATE, DELETE anomalies.
    

|Normal Form|Requirement|
|---|---|
|**1NF**|Atomic (indivisible) column values; no repeating groups.|
|**2NF**|1NF + no partial dependency on a composite PK.|
|**3NF**|2NF + no transitive dependency (non‑key column depends only on PK).|
|**BCNF**|Every determinant is a candidate key (stronger than 3NF).|

### When to Denormalize

- **Read‑Heavy Workloads**: Fewer JOINs, faster reads.
    
- **Reporting / OLAP**: Pre‑aggregated or flattened tables.
    
- **Materialized Views**: Physically store complex aggregations.
    

### Common Interview Queries

- **“Normalize this table to 3NF:”**
    
    |OrderID|CustomerName|Product|Qty|
    |---|---|---|---|
    
- **“What are the drawbacks of over‑normalizing?”**
    
- **“Show how to denormalize for a reporting table.”**
    
- **SQL exercise**:
    
    ```sql
    -- Example: create a denormalized summary table
    CREATE TABLE daily_sales AS
    SELECT
      DATE(order_date) AS sale_date,
      store_id,
      SUM(total_amount) AS total_sales,
      COUNT(*) AS order_count
    FROM orders
    GROUP BY 1,2;
    ```
    

---
---

## 6. Indexing

### Purpose of Indexes

- **Speed up SELECTs** by allowing the database engine to locate rows without scanning the entire table.
    
- **Supporting ORDER BY** and **GROUP BY** operations efficiently when the index matches the sort/group columns.
    

### Types of Indexes

1. **B‑Tree (Balanced Tree)**
    
    - Default in most RDBMS.
        
    - Good for range scans (`WHERE col BETWEEN a AND b`).
        
2. **Hash Index**
    
    - Ideal for exact‐match lookups (`WHERE col = value`), but cannot do range queries.
        
3. **Bitmap Index**
    
    - Very compact; optimal for low‐cardinality columns (e.g., gender or status flags).
        
4. **Clustered vs. Non‑Clustered**
    
    - **Clustered**: Physically orders table data. Only one per table.
        
    - **Non‑Clustered**: Separate structure with pointers back to table rows. Multiple per table.
        
5. **Composite Index**
    
    - Index on multiple columns. Order matters: an index on `(A, B)` can be used for queries filtering on `A` or on both `A` and `B`, but not on `B` alone.
        
6. **Covering Index**
    
    - Includes all columns needed by a query so the engine never has to hit the base table.
        

### Trade‑Offs

- **Pros**: Faster reads, can enforce uniqueness.
    
- **Cons**: Slower writes (INSERT/UPDATE/DELETE must maintain indexes), extra storage, fragmentation over time.
    

### SQL Examples

```sql
-- Single‑column non‑clustered index
CREATE INDEX idx_orders_created_at
  ON orders(created_at);

-- Composite covering index
CREATE INDEX idx_orders_user_date_amt
  ON orders(user_id, order_date)
  INCLUDE (total_amount);

-- Drop an index
DROP INDEX idx_orders_created_at ON orders;
```

### Common Interview Queries

- “Explain the difference between clustered and non‑clustered indexes.”
    
- “When would you use a covering index?”
    
- “How does a composite index work, and why does column order matter?”
    
- “What’s the impact of too many indexes on a table?”
    

---

## 7. Transactions & ACID Properties

### What Is a Transaction?

A **transaction** is a logical unit of work comprising one or more SQL statements that must all succeed or all fail as a group.

### ACID Breakdown

1. **Atomicity**
    
    - All operations in a transaction are treated as a single unit: either all succeed or all are rolled back.
        
2. **Consistency**
    
    - Transactions move the database from one valid state to another, preserving integrity constraints.
        
3. **Isolation**
    
    - Concurrent transactions do not see each other’s intermediate states.
        
4. **Durability**
    
    - Once committed, transaction effects persist even if the system crashes.
        

### How RDBMS Enforces ACID

- **Write‑Ahead Log (WAL)** / **Redo & Undo Logs** for atomicity & durability.
    
- **Locking** or **MVCC** for isolation.
    
- **Integrity checks** at commit time for consistency.
    

### SQL Examples

```sql
-- Explicit transaction block
BEGIN TRANSACTION;

UPDATE accounts
   SET balance = balance - 100
 WHERE id = 101;

UPDATE accounts
   SET balance = balance + 100
 WHERE id = 202;

COMMIT;  -- or ROLLBACK on error
```

### Common Interview Queries

- “What happens if the server crashes after you COMMIT but before data is flushed to disk?”
    
- “How does PostgreSQL’s MVCC model ensure isolation?”
    
- “Explain the difference between logical and physical logging.”
    
- “Why is atomicity important in financial applications?”
    

---

## 8. Concurrency Control & Isolation Levels

### Lock‑Based vs. MVCC

- **Lock‑Based**
    
    - Uses shared (S) and exclusive (X) locks to serialize conflicting accesses.
        
- **MVCC (Multi‑Version Concurrency Control)**
    
    - Writers create new row versions; readers see a snapshot, avoiding read locks.
        

### SQL Standard Isolation Levels

|Level|Dirty Reads|Non‑Repeatable Reads|Phantom Reads|
|---|---|---|---|
|**READ UNCOMMITTED**|Yes|Yes|Yes|
|**READ COMMITTED**|No|Yes|Yes|
|**REPEATABLE READ**|No|No|Yes|
|**SERIALIZABLE**|No|No|No|

### Setting Isolation Levels

```sql
-- PostgreSQL
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- MySQL
SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

### Common Interview Queries

- “Define phantom reads and how SERIALIZABLE prevents them.”
    
- “What’s the difference between optimistic and pessimistic locking?”
    
- “How does snapshot isolation differ from SERIALIZABLE?”
    
- “Give an example of a lost‑update anomaly and how to avoid it.”
    

---

## 9. Deadlocks & Resolution Strategies

### What Is a Deadlock?

A deadlock occurs when two or more transactions are each waiting for locks held by the other, creating a cycle and halting progress.

### Detection & Resolution

- **Wait‑For Graph**: DBMS builds a graph of transactions vs. locks; cycles indicate deadlocks.
    
- **Victim Selection**: One transaction is chosen as the “victim” and rolled back to break the cycle.
    

### Prevention Techniques

- **Lock Ordering**: Always acquire locks in a predefined order.
    
- **Timeouts**: Abort transactions that wait too long.
    
- **No‑Wait Policy**: Immediately abort if a lock cannot be acquired.
    

### Monitoring in MySQL / PostgreSQL

```sql
-- MySQL: view InnoDB transactions & locks
SHOW ENGINE INNODB STATUS\G;

-- PostgreSQL: list blocked queries
SELECT blocked_locks.pid AS blocked_pid,
       blocking_locks.pid AS blocking_pid,
       blocked_activity.query AS blocked_query,
       blocking_activity.query AS blocking_query
  FROM pg_locks blocked_locks
  JOIN pg_locks blocking_locks
    ON blocked_locks.locktype = blocking_locks.locktype
   AND blocked_locks.locktype = blocking_locks.locktype
   AND blocked_locks.database IS NOT DISTINCT FROM blocking_locks.database
   AND blocked_locks.relation IS NOT DISTINCT FROM blocking_locks.relation
   AND blocked_locks.page IS NOT DISTINCT FROM blocking_locks.page
   AND blocked_locks.tuple IS NOT DISTINCT FROM blocking_locks.tuple
  JOIN pg_stat_activity blocked_activity
    ON blocked_activity.pid = blocked_locks.pid
  JOIN pg_stat_activity blocking_activity
    ON blocking_activity.pid = blocking_locks.pid
 WHERE NOT blocked_locks.granted
   AND blocking_locks.granted;
```

### Common Interview Queries

- “How does your DBMS detect deadlocks?”
    
- “Explain the difference between a deadlock and a lock wait timeout.”
    
- “Describe a scenario leading to a phantom deadlock.”
    
- “What strategies would you implement to minimize deadlocks in a high‑throughput system?”
    

---

## 10. Joins & Set Operations

### Types of Joins

- **INNER JOIN**: Returns rows where matching keys exist in both tables.
    
- **LEFT/RIGHT OUTER JOIN**: Returns all rows from one side, plus matches from the other.
    
- **FULL OUTER JOIN**: Combines LEFT and RIGHT—keeps all rows from both.
    
- **CROSS JOIN**: Cartesian product of the two tables (rarely used).
    
- **SELF JOIN**: A table joined with itself (often via aliases).
    

### Join Algorithms

1. **Nested Loop Join**
    
2. **Hash Join**
    
3. **Merge Join**
    

### Set Operations

- **UNION / UNION ALL**: Combine result sets (UNION removes duplicates).
    
- **INTERSECT**: Rows common to both queries.
    
- **EXCEPT / MINUS**: Rows in first query not in second.
    

### SQL Examples

```sql
-- INNER JOIN
SELECT u.name, o.total_amount
  FROM users u
  JOIN orders o ON u.id = o.user_id;

-- LEFT OUTER JOIN
SELECT r.name, m.item_name
  FROM restaurants r
  LEFT JOIN menu m ON r.id = m.restaurant_id;

-- UNION vs. UNION ALL
SELECT city FROM users
UNION
SELECT city FROM restaurants;

SELECT city FROM users
UNION ALL
SELECT city FROM restaurants;

-- INTERSECT example (PostgreSQL)
SELECT user_id FROM orders_2024
INTERSECT
SELECT user_id FROM orders_2025;
```

### Common Interview Queries

- “What’s the difference between INNER JOIN and CROSS JOIN?”
    
- “When would you choose a hash join over a nested loop?”
    
- “Explain how UNION ALL differs from UNION in performance.”
    
- “How do you eliminate duplicate rows after a join?”
    

---
# Solved Questions
---

### 1. Schema Design & Complex Query

**Q:** Design a minimal schema to store **users**, **restaurants**, **orders**, and **order_items**. Then write a SQL query to list the **top 5 restaurants by total revenue** in the **last 30 days**, including the revenue amount.

```sql
-- 1. Schema
CREATE TABLE users (
  user_id     BIGINT PRIMARY KEY,
  name        VARCHAR(100),
  signup_date DATE
);

CREATE TABLE restaurants (
  restaurant_id BIGINT PRIMARY KEY,
  name          VARCHAR(150),
  city          VARCHAR(50)
);

CREATE TABLE orders (
  order_id       BIGINT PRIMARY KEY,
  user_id        BIGINT NOT NULL REFERENCES users(user_id),
  restaurant_id  BIGINT NOT NULL REFERENCES restaurants(restaurant_id),
  order_date     TIMESTAMP NOT NULL,
  total_amount   DECIMAL(10,2)
);

CREATE TABLE order_items (
  order_id    BIGINT NOT NULL REFERENCES orders(order_id),
  item_id     BIGINT NOT NULL,
  quantity    INT,
  price       DECIMAL(10,2),
  PRIMARY KEY (order_id, item_id)
);

-- 2. Top‑5 restaurants by revenue in last 30 days
SELECT
  r.restaurant_id,
  r.name AS restaurant_name,
  SUM(o.total_amount) AS revenue
FROM orders o
JOIN restaurants r ON o.restaurant_id = r.restaurant_id
WHERE o.order_date >= NOW() - INTERVAL '30 days'
GROUP BY r.restaurant_id, r.name
ORDER BY revenue DESC
LIMIT 5;
```

**Explanation:**

- We separate line‐items (`order_items`) from the order header (`orders`) to allow multiple items per order.
    
- We store `total_amount` on `orders` to avoid aggregating `order_items` in every query (denormalization for performance).
    
- The query groups by restaurant, sums `total_amount`, filters by date with an index on `order_date`, and fetches the top 5.
    

---

### 2. Window Functions for “Top N per Group”

**Q:** For each restaurant, find its **highest‑grossing day** (the date with the maximum total order value) over the past quarter.

```sql
WITH daily_rev AS (
  SELECT
    restaurant_id,
    DATE(order_date) AS day,
    SUM(total_amount) AS revenue
  FROM orders
  WHERE order_date >= NOW() - INTERVAL '3 months'
  GROUP BY restaurant_id, DATE(order_date)
),
ranked_rev AS (
  SELECT
    restaurant_id,
    day,
    revenue,
    ROW_NUMBER() OVER (
      PARTITION BY restaurant_id
      ORDER BY revenue DESC
    ) AS rn
  FROM daily_rev
)
SELECT
  r.restaurant_id,
  r.name            AS restaurant_name,
  dr.day            AS top_day,
  dr.revenue
FROM ranked_rev dr
JOIN restaurants r ON dr.restaurant_id = r.restaurant_id
WHERE dr.rn = 1;
```

**Explanation:**

- First CTE computes daily revenue per restaurant.
    
- Second CTE ranks days per restaurant by revenue using `ROW_NUMBER()`.
    
- Final query picks the top (`rn=1`) day for each restaurant and joins to get the name.
    

---

### 3. Query Optimization & Indexing

**Q:** You notice this query is slow. How would you optimize it?

```sql
SELECT u.user_id, u.name, COUNT(o.order_id) AS num_orders
FROM users u
LEFT JOIN orders o
  ON u.user_id = o.user_id
  AND o.order_date >= '2025-07-01'
GROUP BY u.user_id, u.name
HAVING COUNT(o.order_id) > 100;
```

1. **Add a composite index** on `(user_id, order_date)` in `orders` so the join/filter is index‑supported:
    
    ```sql
    CREATE INDEX idx_orders_user_date
      ON orders(user_id, order_date);
    ```
    
2. **Rewrite HAVING as WHERE** on a subquery to filter early:
    
    ```sql
    SELECT u.user_id, u.name, o.num_orders
    FROM users u
    JOIN (
      SELECT user_id, COUNT(*) AS num_orders
      FROM orders
      WHERE order_date >= '2025-07-01'
      GROUP BY user_id
      HAVING COUNT(*) > 100
    ) o ON u.user_id = o.user_id;
    ```
    
3. **Explain Plans**: Always check `EXPLAIN ANALYZE` to confirm index usage and avoid full table scans.
    

**Why:**

- The composite index supports both the join key and the date filter.
    
- Pushing filtering into a subquery reduces the number of rows joined to `users`.
    

---

### 4. Handling High‑Volume Writes with Partitioning

**Q:** Orders can exceed 10 million records per month. How would you partition the `orders` table to maintain performance?

- **Range Partitioning** by **month** on `order_date`:
    
    ```sql
    -- PostgreSQL example
    CREATE TABLE orders_y2025m01 PARTITION OF orders
      FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
    CREATE TABLE orders_y2025m02 PARTITION OF orders
      FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
    -- ... and so on
    ```
    
- **Benefits**:
    
    - Queries on recent data hit only a few partitions.
        
    - Old partitions can be **detached** and **archived** quickly.
        
- **Indexing**: Create indexes on each partition (or use `CREATE INDEX … ON ONLY orders` to propagate).
    
- **Maintenance**: Automate partition creation via a scheduled job.
    

---

### 5. Concurrency & Idempotent Status Updates

**Q:** Multiple services may update an order’s `status` concurrently (e.g., “placed” → “confirmed” → “delivered”). How do you ensure no status transitions are lost or applied out of order?

1. **Use a check constraint** or **trigger** to enforce valid transitions:
    
    ```sql
    ALTER TABLE orders
      ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'placed';
    
    CREATE FUNCTION enforce_status_transition() RETURNS trigger AS $$
    BEGIN
      IF (OLD.status, NEW.status) NOT IN (
        ('placed','confirmed'),
        ('confirmed','out_for_delivery'),
        ('out_for_delivery','delivered')
      ) THEN
        RAISE EXCEPTION 'Invalid status transition: % → %', OLD.status, NEW.status;
      END IF;
      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    
    CREATE TRIGGER trg_status_transition
      BEFORE UPDATE OF status ON orders
      FOR EACH ROW EXECUTE FUNCTION enforce_status_transition();
    ```
    
2. **Optimistic Locking**: Add a `version` column and require `WHERE version = :old_version` in the `UPDATE`, incrementing `version`. If zero rows updated, retry.
    
3. **Idempotency Tokens**: For external API calls, use an `idempotency_key` so retries don’t duplicate transitions.
    

**Why:**

- **DB‑level enforcement** ensures no invalid or out‑of‑order transition ever persists.
    
- **Optimistic locking** prevents lost updates without heavy locking overhead.
    

---

### 6. Real‑Time Analytics with Materialized Views

**Q:** You need to serve a dashboard showing **hourly order volume** for the last 24 hours, refreshed every minute. How would you build this efficiently?

1. **Create a rolling materialized view** that aggregates orders by hour:
    
    ```sql
    CREATE MATERIALIZED VIEW hourly_order_stats AS
    SELECT
      date_trunc('hour', order_date) AS hour,
      COUNT(*) AS order_count
    FROM orders
    WHERE order_date >= NOW() - INTERVAL '1 day'
    GROUP BY 1;
    ```
    
2. **Refresh incrementally** using `CONCURRENTLY` and limiting to the newest hour:
    
    ```sql
    -- Cron job running every minute:
    REFRESH MATERIALIZED VIEW CONCURRENTLY hourly_order_stats;
    ```
    
3. **Index** the view on `hour` for fast lookups:
    
    ```sql
    CREATE INDEX idx_hourly_stats_hour ON hourly_order_stats(hour);
    ```
    
4. **Serve Dashboard** from this view rather than live-scanning the 10M rows/hour.
    

**Why:**

- Materialized view avoids repeated full scans.
    
- `CONCURRENTLY` allows reads during refresh.
    
- Hourly granularity keeps the data small (~24 rows).
    

---
# More

---

### 7. Designing for High Availability & Automatic Failover

**Q:** Zomato must stay online even if one database node crashes. Design a replication and failover strategy for the primary **orders** database.

1. **Asynchronous Master–Slave Replication**
    
    ```text
    – Primary (“master”) accepts writes.
    – Two or more secondaries (“slaves”) replicate the WAL stream.
    – Slaves are read‑only and used for analytics/BI.
    ```
    
2. **Automated Failover with Replica Promotion**
    
    - Deploy **Patroni** (PostgreSQL) or **MHA** (MySQL) + **etcd/ZooKeeper** for leader election.
        
    - Upon master failure:
        
        1. Health checks detect unresponsive master.
            
        2. Orchestrator promotes the most up‑to‑date slave.
            
        3. Clients reconnect via a virtual IP or service discovery.
            
3. **Configuration Snippet (PostgreSQL + Patroni)**
    
    ```yaml
    scope: orders_db
    namespace: /service/
    name: master
    
    restapi:
      listen: 0.0.0.0:8008
      connect_address: 10.0.0.1:8008
    
    etcd:
      host: 10.0.0.10:2379
    
    bootstrap:
      dcs:
        ttl: 30
        loop_wait: 10
        retry_timeout: 10
        maximum_lag_on_failover: 1048576
      initdb:
        - encoding: UTF8
        - data-checksums
    
    postgresql:
      listen: 0.0.0.0:5432
      connect_address: 10.0.0.1:5432
      data_dir: /var/lib/postgresql/data
      parameters:
        wal_level: replica
        max_wal_senders: 10
        wal_keep_segments: 64
    ```
    
4. **Client‑Side Configuration**
    
    - Use a **connection pooler** (PgBouncer/MySQL Proxy) that automatically routes writes to master and reads to slaves.
        
    - On failover, update the pooler’s target master via the orchestration layer.
        

**Why This Works:**

- Asynchronous replication keeps slaves almost up‑to‑date with minimal write latency impact.
    
- Patroni’s automatic promotion and health checks eliminate manual intervention.
    
- Connection pooling abstracts failover logic away from the application.
    

---

### 8. Balancing Consistency, Availability & Partition Tolerance (CAP)

**Q:** Zomato’s search service must return restaurant data quickly even if a regional datacenter is unreachable. How do you apply CAP principles when choosing your database and consistency model?

1. **Identify Requirements**
    
    - **Availability (A):** Search queries must still work when a DC is down.
        
    - **Partition Tolerance (P):** Network splits between regions can happen.
        
    - **Consistency (C):** Some stale data is acceptable for search results.
        
2. **Choose AP‑oriented System**
    
    - **Cassandra** or **DynamoDB** with **eventual consistency**.
        
    - Configure **read/write quorum** to tune freshness vs. latency:
        
        - Write to **W=1** replica (fast writes).
            
        - Read from **R=1** replica (fast reads).
            
        - **W + R > Replication Factor** for stronger consistency when needed.
            
3. **Consistency Tuning Example (Cassandra CQL)**
    
    ```sql
    -- Create a keyspace replicated across 3 DCs
    CREATE KEYSPACE zomato_search WITH
      replication = {
        'class': 'NetworkTopologyStrategy',
        'dc1': '3',  -- 3 replicas in primary DC
        'dc2': '2'   -- 2 replicas in secondary DC
      };
    
    -- Write with QUORUM consistency
    INSERT INTO restaurant_data (...)
      USING CONSISTENCY QUORUM
      VALUES (...);
    
    -- Read with LOCAL_QUORUM for slightly fresher, region‑local data
    SELECT * FROM restaurant_data
      WHERE id = '1234'
      USING CONSISTENCY LOCAL_QUORUM;
    ```
    
4. **Why Eventual Consistency Works for Search**
    
    - Minor staleness (a few seconds) is acceptable for search indexes.
        
    - High availability and partition tolerance ensure that ANY replica can serve queries.
        

**Key Takeaway:**

- Sacrifice **strict** consistency for **availability** and **partition tolerance** in read‑heavy, latency‑sensitive services like search.
    

---

### 9. Caching Layer for Hot Data

**Q:** How would you integrate a cache to accelerate reads of popular restaurant profiles and minimize DB load?

1. **Deploy Redis (in‑memory KV store)**
    
    - **TTL** of e.g. 5 minutes to keep data reasonably fresh.
        
    - **Cache Aside** pattern in the application:
        
        ```pseudo
        function getRestaurant(id):
          if exists in redis.get(id):
            return cached value
          else:
            data = db.query("SELECT * FROM restaurants WHERE id=?", id)
            redis.set(id, data, ttl=300)
            return data
        ```
        
2. **Handle Cache Invalidation**
    
    - **Time-based TTL** handles most cases.
        
    - **Explicit Invalidation** on updates:
        
        ```sql
        -- After updating the restaurant row:
        redis.del(restaurant_id)
        ```
        
3. **Mitigate Cache Stampede**
    
    - Use **mutex locks** or **request coalescing** so that only one request fetches from the DB on a cache miss.
        
4. **Monitoring & Metrics**
    
    - Track **cache hit ratio** (> 90% ideal).
        
    - Alert if miss ratio spikes (could indicate TTL too low or invalidation bug).
        

**Why This Works:**

- Redis drastically reduces p99 read latency and offloads traffic from the DB cluster.
    
- TTL plus explicit invalidation keeps the cache coherent.
    

---

### 10. Using Change Data Capture (CDC) for Analytics Pipelines

**Q:** Zomato wants a near‑real‑time analytics pipeline without hitting the primary DB. How do you stream order events into a data warehouse?

1. **Enable CDC on Primary**
    
    - **PostgreSQL** with **wal2json** output plugin or **Debezium** connector.
        
2. **Stream to Kafka**
    
    - Debezium publishes every `INSERT`/`UPDATE`/`DELETE` on **orders** to a Kafka topic (`orders.cdc`).
        
3. **Consume and Load**
    
    - Use **Kafka Connect** with the **JDBC Sink** to load into your analytics DB (e.g., BigQuery, Redshift, ClickHouse) in micro‑batches.
        
4. **Schema Evolution**
    
    - Debezium auto‑detects schema changes and can update Kafka schema registry entries.
        
5. **Minimal Impact on OLTP**
    
    - CDC reads WAL asynchronously; no extra locks on the primary.
        

**Architecture Diagram (simplified):**

```
Primary DB  ── WAL stream ──> Debezium ──> Kafka ──> Kafka Connect ──> Data Warehouse
```

**Why CDC?**

- Gives near‑real‑time data in analytics without modifying application writes.
    
- Decouples OLTP and OLAP workloads entirely.
    

---

### 11. Designing a Geo‑Distributed “Nearest Restaurant” Query

**Q:** How would you design a schema and indexes to efficiently find the **10 nearest restaurants** to a given GPS coordinate in each city?

1. **Store Coordinates**
    
    ```sql
    ALTER TABLE restaurants
      ADD COLUMN latitude  DOUBLE PRECISION NOT NULL,
      ADD COLUMN longitude DOUBLE PRECISION NOT NULL;
    ```
    
2. **Use a Spatial Index**
    
    - **PostGIS** extension on PostgreSQL:
        
        ```sql
        -- Add a Geography column
        ALTER TABLE restaurants
          ADD COLUMN geog GEOGRAPHY(Point, 4326);
        
        -- Populate it
        UPDATE restaurants
          SET geog = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326);
        
        -- Create a GiST index
        CREATE INDEX idx_restaurants_geog
          ON restaurants USING GIST(geog);
        ```
        
3. **Nearest‑Neighbor Query**
    
    ```sql
    SELECT id, name,
           ST_Distance(geog, ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)) AS dist_m
      FROM restaurants
     WHERE city = :city
     ORDER BY geog <-> ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)
     LIMIT 10;
    ```
    
    - The `<->` operator invokes index‑accelerated KNN search on the GiST index.
        
4. **Sharding by City**
    
    - If a city has millions of restaurants, maintain separate tables or partitions per city to keep index size manageable.
        

**Why This Works:**

- Spatial indexing reduces a full‑table scan into an indexed K‑NN search.
    
- Partitioning/sharding ensures low per‑index latency even at global scale.
    

---

### 12. Multi‑Master Replication & Conflict Resolution

**Q:** Zomato runs writes in multiple datacenters (multi‑master). How do you handle write conflicts and ensure data convergence?

1. **Choose a Conflict‑Resolution Strategy**
    
    - **Last‑Write‑Wins (LWW):** Compare timestamps; highest “last_modified” wins.
        
    - **Custom Merge Function:** E.g., sum quantities, pick max rating, or merge JSON blobs.
        
2. **Implementation in Cassandra**
    
    ```cql
    -- Table with a write_time-based LWW
    CREATE TABLE restaurant_stats (
      restaurant_id UUID,
      metric        text,
      value         bigint,
      PRIMARY KEY (restaurant_id, metric)
    )
      WITH default_time_to_live = 0
      AND compaction = { 'class': 'LeveledCompactionStrategy' };
    ```
    
    - Cassandra’s built‑in LWW uses CQL `WRITETIME(value)` to pick latest.
        
3. **Custom Resolver in DynamoDB**
    
    ```jsonc
    // Define an AWS Lambda for DynamoDB Streams
    exports.handler = async (event) => {
      for (const record of event.Records) {
        if (record.eventName === 'MODIFY') {
          const old = record.dynamodb.OldImage;
          const neu = record.dynamodb.NewImage;
          // Example: choose higher order_count
          const resolved = (neu.order_count.N > old.order_count.N)
            ? neu.order_count.N
            : old.order_count.N;
          // Write back the resolved value if different
          // ...
        }
      }
    };
    ```
    
4. **Ensure Convergence**
    
    - Use **vector clocks** or **Lamport timestamps** if causal ordering matters.
        
    - Periodically run a **background reconciliation** job to detect and merge any divergent rows.
        

**Why This Works:**

- LWW is simple and performant when slight overwrites are acceptable.
    
- Custom resolvers handle domain‑specific merges (e.g., accumulating metrics).
    
- Periodic reconciliation guarantees eventual consistency across sites.
    

---

### 13. Zero‑Downtime Schema Migrations

**Q:** You need to add a NOT NULL column `rating_reviewed` to a 1 billion‑row `reviews` table without blocking reads/writes. Outline a safe migration.

1. **Add Nullable Column**
    
    ```sql
    ALTER TABLE reviews ADD COLUMN rating_reviewed BOOLEAN;
    ```
    
2. **Backfill in Batches**
    
    ```sql
    -- Pseudo‑script running in chunks of 1M rows
    UPDATE reviews
       SET rating_reviewed = TRUE
     WHERE rating_reviewed IS NULL
       AND id BETWEEN :start AND :end;
    ```
    
3. **Add NOT NULL Constraint** (disable validation)
    
    ```sql
    ALTER TABLE reviews
      ALTER COLUMN rating_reviewed SET NOT NULL
      NOT VALID;
    ```
    
4. **Validate Constraint**
    
    ```sql
    ALTER TABLE reviews VALIDATE CONSTRAINT reviews_rating_reviewed_not_null;
    ```
    
5. **Cleanup**
    
    - Drop any temporary flags or triggers used during backfill.
        

**Why This Works:**

- Adding nullable column is instantaneous.
    
- Batching avoids long‑running locks.
    
- Deferring validation (`NOT VALID`) prevents table rewrite.
    
- Final validation is fast if backfill is complete.
    

---

### 14. High‑Volume Bulk Ingestion & ETL

**Q:** You must load 50 million rows of historical order data nightly into the OLAP cluster. Describe an efficient ETL approach.

1. **Staging via Flat Files**
    
    - Export data from OLTP → compressed CSV/Parquet on HDFS/S3.
        
2. **Parallel Loader**
    
    - Use **COPY** (Postgres), **LOAD DATA INFILE** (MySQL), or **COPY INTO** (Snowflake) with multiple file slices:
        
        ```sql
        COPY INTO analytics.orders_history
        FROM @s3://zomato/historical/orders/
        FILE_FORMAT = (TYPE=PARQUET)
        ON_ERROR = 'CONTINUE';
        ```
        
3. **Partition‑Aware Load**
    
    - Target partitions by date to enable pruning:
        
        ```sql
        COPY INTO analytics.orders_y2020m01
        ...
        ```
        
4. **Incremental Upserts**
    
    - Use **MERGE** to insert new and update changed rows:
        
        ```sql
        MERGE INTO analytics.orders_history tgt
        USING staging.orders_updates src
        ON tgt.order_id = src.order_id
        WHEN MATCHED THEN UPDATE SET ...
        WHEN NOT MATCHED THEN INSERT (...);
        ```
        
5. **Orchestration & Monitoring**
    
    - Schedule via Airflow with SLA alerts.
        
    - Verify row counts and checksum hashes post‑load.
        

**Why This Works:**

- Staging separates ETL from OLTP, preventing load spikes.
    
- File‑based bulk load is orders of magnitude faster than individual inserts.
    
- Partition‑aware ingestion speeds downstream queries.
    

---

### 15. Data Archival & Purging Strategy

**Q:** Zomato must keep only 2 years of orders online. Design an archival and purge process that’s safe and reversible.

1. **Archive to Cold Storage**
    
    ```sql
    CREATE TABLE orders_archive PARTITION OF orders
      FOR VALUES FROM ('2023-01-01') TO ('2024-01-01')
      WITH (storage_parameters);
    -- Detach and export partition to object storage:
    ALTER TABLE orders_archive DETACH PARTITION;
    \COPY orders_archive TO 's3://zomato/archive/orders_2023.parquet' PARQUET;
    ```
    
2. **Verify & Delete**
    
    - Confirm integrity (row counts, checksums) in the archive.
        
    - Drop the partition from the live cluster:
        
        ```sql
        DROP TABLE orders_archive;
        ```
        
3. **Reversible Restore**
    
    - To restore, **COPY** back from S3 into a new partition and **ATTACH**:
        
        ```sql
        CREATE TABLE orders_archive (...);
        \COPY orders_archive FROM 's3://zomato/archive/orders_2023.parquet' PARQUET;
        ALTER TABLE orders ATTACH PARTITION orders_archive FOR VALUES ...;
        ```
        
4. **Automation**
    
    - Schedule with cron or Airflow:
        
        - Monthly job: archive oldest month → drop.
            
        - Weekly verification checksums.
            

**Why This Works:**

- Partition‑based archive/purge is constant‑time per partition.
    
- Detached export ensures minimal impact on live workload.
    
- Archival in Parquet is both compact and queryable if needed.
    

---

### 16. Querying JSON & Semi‑Structured Data

**Q:** The `menu` table stores item details in a JSON column `specs`. Write a query to extract all items where `specs.ingredients` array contains “chili”.

1. **Assume Table Definition**
    
    ```sql
    CREATE TABLE menu (
      item_id   BIGINT PRIMARY KEY,
      name      VARCHAR(100),
      specs     JSONB
    );
    ```
    
2. **Index the JSON Path**
    
    ```sql
    CREATE INDEX idx_menu_specs_ingredients
      ON menu USING GIN ((specs -> 'ingredients'));
    ```
    
3. **Query with JSONB Operators**
    
    ```sql
    SELECT item_id, name, specs
      FROM menu
     WHERE specs -> 'ingredients' ? 'chili';
    ```
    
4. **Extract Nested Values**
    
    ```sql
    SELECT
      item_id,
      specs ->> 'calories' AS calories_per_serving
    FROM menu
    WHERE specs -> 'ingredients' ? 'chili';
    ```
    

**Why This Works:**

- GIN index on the array field makes the “contains” test efficient.
    
- JSONB operators (`?`, `->`, `->>`) allow direct path queries without full scans.
    
