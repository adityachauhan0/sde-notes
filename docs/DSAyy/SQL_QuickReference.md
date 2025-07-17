
## 🛠️ Setup & Connection

```sql
-- Connect (CLI)
mysql -u username -p

-- Create Database
CREATE DATABASE db_name;

-- Use Database
USE db_name;

-- Show Databases / Tables
SHOW DATABASES;
SHOW TABLES;
```

---

## 📦 Table Operations

```sql
-- Create Table
CREATE TABLE table_name (
  id INT PRIMARY KEY AUTO_INCREMENT,
  col1 VARCHAR(255) NOT NULL,
  col2 INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Modify Table
ALTER TABLE table_name ADD column_name DATATYPE;
ALTER TABLE table_name DROP COLUMN column_name;
ALTER TABLE table_name MODIFY column_name DATATYPE;
ALTER TABLE table_name RENAME TO new_table_name;

-- Drop Table
DROP TABLE table_name;

-- Truncate Table (Delete all rows)
TRUNCATE TABLE table_name;

-- Rename Column (MySQL 8+)
ALTER TABLE table_name RENAME COLUMN old_name TO new_name;
```

---

## 📥 CRUD Operations

```sql
-- INSERT
INSERT INTO table_name (col1, col2) VALUES ('val1', 123);

-- INSERT MULTIPLE
INSERT INTO table_name (col1, col2) VALUES ('a', 1), ('b', 2);

-- SELECT
SELECT * FROM table_name;
SELECT col1, col2 FROM table_name WHERE col2 > 10;

-- UPDATE
UPDATE table_name SET col1 = 'new' WHERE id = 1;

-- DELETE
DELETE FROM table_name WHERE id = 1;
```

---

## 🧰 Query Options

```sql
-- Aliases
SELECT col1 AS alias FROM table_name;

-- WHERE Operators
=, !=, <>, >, <, >=, <=, BETWEEN, IN (...), NOT IN (...), IS NULL, IS NOT NULL, LIKE, REGEXP

-- Logical
AND, OR, NOT

-- LIMIT & OFFSET
SELECT * FROM table LIMIT 10 OFFSET 5;

-- ORDER
SELECT * FROM table ORDER BY col1 ASC, col2 DESC;

-- GROUP BY
SELECT col1, COUNT(*) FROM table GROUP BY col1;

-- HAVING
SELECT col1, COUNT(*) FROM table GROUP BY col1 HAVING COUNT(*) > 1;
```

---

## 🔗 Joins

```sql
-- INNER JOIN
SELECT * FROM a INNER JOIN b ON a.id = b.a_id;

-- LEFT JOIN
SELECT * FROM a LEFT JOIN b ON a.id = b.a_id;

-- RIGHT JOIN
SELECT * FROM a RIGHT JOIN b ON a.id = b.a_id;

-- FULL OUTER JOIN (emulated)
SELECT * FROM a LEFT JOIN b ON a.id = b.a_id
UNION
SELECT * FROM a RIGHT JOIN b ON a.id = b.a_id;
```

---

## 🔄 Subqueries & Unions

```sql
-- Subquery
SELECT * FROM table WHERE id IN (SELECT id FROM another WHERE col > 5);

-- UNION
SELECT col1 FROM a
UNION
SELECT col1 FROM b;
```

---

## 🧠 Indexes & Keys

```sql
-- Create Index
CREATE INDEX idx_name ON table(col);

-- Unique Index
CREATE UNIQUE INDEX idx_name ON table(col);

-- Drop Index
DROP INDEX idx_name ON table;

-- Foreign Key
ALTER TABLE child ADD FOREIGN KEY (col) REFERENCES parent(id);
```

---

## ⚙️ Constraints

```sql
-- NOT NULL, UNIQUE, PRIMARY KEY, DEFAULT, CHECK
CREATE TABLE t (
  id INT PRIMARY KEY,
  email VARCHAR(255) UNIQUE,
  age INT CHECK (age >= 18)
);
```

---

## 📊 Aggregate Functions

```sql
SELECT COUNT(*), AVG(col), MIN(col), MAX(col), SUM(col) FROM table;
```

---

## 🕐 Date & Time

```sql
-- Now
SELECT NOW(), CURDATE(), CURTIME();

-- Extract parts
SELECT YEAR(date_col), MONTH(date_col), DAY(date_col);

-- Date arithmetic
SELECT DATE_ADD(NOW(), INTERVAL 7 DAY);
SELECT DATEDIFF(NOW(), '2023-01-01');
```

---

## 🔧 User & Privileges

```sql
-- Create user
CREATE USER 'user'@'localhost' IDENTIFIED BY 'pass';

-- Grant privileges
GRANT ALL PRIVILEGES ON db.* TO 'user'@'localhost';

-- Revoke
REVOKE ALL PRIVILEGES ON db.* FROM 'user'@'localhost';

-- Show privileges
SHOW GRANTS FOR 'user'@'localhost';
```

---

## 🧹 Transactions

```sql
START TRANSACTION;
-- SQL Statements
COMMIT;
-- or
ROLLBACK;
```

---

## 🧪 Useful Admin Queries

```sql
-- Show columns
DESCRIBE table_name;

-- Show processlist
SHOW PROCESSLIST;

-- Show variables / status
SHOW VARIABLES;
SHOW STATUS;
```

---

## 🔗 JOIN Examples

### 1. INNER JOIN

```sql
SELECT e.name, d.department_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;
```

### 2. LEFT JOIN

```sql
SELECT e.name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;
```

### 3. RIGHT JOIN

```sql
SELECT e.name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;
```

### 4. FULL OUTER JOIN (Emulated)

```sql
SELECT e.name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id
UNION
SELECT e.name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;
```

### 5. SELF JOIN

```sql
SELECT e1.name AS employee, e2.name AS manager
FROM employees e1
JOIN employees e2 ON e1.manager_id = e2.id;
```

---

## 🧠 Normalization (Theory)

### **1NF (First Normal Form)**

- Atomic columns (no repeating groups).
    
- Example: Split comma-separated values into rows.
    

### **2NF (Second Normal Form)**

- 1NF + No partial dependency (all non-key columns depend on the whole primary key).
    

### **3NF (Third Normal Form)**

- 2NF + No transitive dependency (non-key column depends only on the key).
    

### **BCNF (Boyce-Codd Normal Form)**

- Every determinant is a candidate key.
    

✅ _In practice: Use up to 3NF unless performance needs denormalization._

---

## 📌 Constraints & Keys

### 1. PRIMARY KEY

```sql
id INT PRIMARY KEY
```

### 2. FOREIGN KEY

```sql
FOREIGN KEY (dept_id) REFERENCES departments(id)
```

### 3. UNIQUE

```sql
email VARCHAR(255) UNIQUE
```

### 4. CHECK

```sql
CHECK (salary >= 0)
```

### 5. DEFAULT

```sql
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

---

## 🔄 Common Patterns

### 1. Top N Records

```sql
SELECT * FROM employees ORDER BY salary DESC LIMIT 5;
```

### 2. Duplicate Detection

```sql
SELECT email, COUNT(*) 
FROM users 
GROUP BY email 
HAVING COUNT(*) > 1;
```

### 3. NULL Handling

```sql
SELECT IFNULL(phone, 'N/A') FROM users;
```

### 4. Case Statement

```sql
SELECT name,
  CASE 
    WHEN salary > 50000 THEN 'High'
    ELSE 'Low'
  END AS salary_level
FROM employees;
```

### 5. Subquery in WHERE

```sql
SELECT * FROM orders
WHERE customer_id IN (
  SELECT id FROM customers WHERE city = 'London'
);
```

---

## 📐 Table Normalization Example

**Unnormalized Table:**

```sql
-- Bad table: multiple values in one field
orders(order_id, customer, products) 
-- e.g., products = 'TV,Phone'
```

**Normalized (3NF):**

```sql
customers(customer_id, name)
orders(order_id, customer_id)
products(product_id, name)
order_products(order_id, product_id)
```

---

## 🪜 Window Functions (MySQL 8+)

### 1. RANK, ROW_NUMBER, DENSE_RANK

```sql
SELECT name, salary,
  RANK() OVER (ORDER BY salary DESC) AS rank
FROM employees;
```

### 2. Running Total

```sql
SELECT name, salary,
  SUM(salary) OVER (ORDER BY name) AS running_total
FROM employees;
```

---

## 🔎 CTEs (Common Table Expressions)

### 1. Basic CTE

```sql
WITH high_earners AS (
  SELECT * FROM employees WHERE salary > 100000
)
SELECT * FROM high_earners;
```

### 2. Recursive CTE

```sql
WITH RECURSIVE numbers AS (
  SELECT 1 AS n
  UNION ALL
  SELECT n + 1 FROM numbers WHERE n < 5
)
SELECT * FROM numbers;
```

---

## 📊 Interview-Style Query Snippets

### 1. Second Highest Salary

```sql
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

### 2. Employees in All Departments (Division)

```sql
SELECT e.name
FROM employees e
WHERE NOT EXISTS (
  SELECT d.id FROM departments d
  WHERE NOT EXISTS (
    SELECT * FROM works w WHERE w.emp_id = e.id AND w.dept_id = d.id
  )
);
```

### 3. Find Duplicate Rows

```sql
SELECT col, COUNT(*) 
FROM table
GROUP BY col
HAVING COUNT(*) > 1;
```

---
# Normalization Quesitons
## ✅ **1. Orders Table with Repeating Groups**

### 📌 Unnormalized Table:

```
Orders(order_id, customer_name, product1, product2, product3)
```

### 🚫 Problem:

- Repeating groups (`product1`, `product2`, `product3`) violate **1NF**.
    

---

### ✅ 1NF:

- Split repeating groups into individual rows.
    

```sql
Orders(order_id, customer_name, product_name)
```

|order_id|customer_name|product_name|
|---|---|---|
|1|Alice|TV|
|1|Alice|Phone|
|1|Alice|Laptop|

---

### ✅ 2NF:

- Assume `order_id` is the primary key. All non-key attributes depend on it → already 2NF.
    

---

### ✅ 3NF:

- All attributes are atomic, and no transitive dependencies exist. **Already in 3NF**.
    

---

## ✅ **2. Employee Skills Table**

### 📌 Unnormalized Table:

```
Employees(emp_id, emp_name, skills)
-- Example: skills = "Python, Java, SQL"
```

### 🚫 Problem:

- `skills` is multi-valued → violates **1NF**.
    

---

### ✅ 1NF:

```sql
EmployeeSkills(emp_id, emp_name, skill)
```

|emp_id|emp_name|skill|
|---|---|---|
|1|John|Python|
|1|John|Java|
|1|John|SQL|

---

### ✅ 2NF:

- Assuming `emp_id` is the primary key → already 2NF.
    

---

### ✅ 3NF:

- `emp_name` depends on `emp_id`, and `skill` is independent.  
    To normalize further:
    

**Split into:**

```sql
Employees(emp_id, emp_name)
Skills(emp_id, skill)
```

---

## ✅ **3. Enrollments Table with Partial Dependency**

### 📌 Unnormalized Table:

```
Enrollments(student_id, course_id, course_name, instructor)
```

### 🚫 Problem:

- Composite key: (student_id, course_id)
    
- `course_name` and `instructor` depend only on `course_id` → **partial dependency** → violates **2NF**.
    

---

### ✅ 1NF:

Already in 1NF (all fields atomic).

---

### ✅ 2NF:

Break into two tables:

```sql
Enrollments(student_id, course_id)
Courses(course_id, course_name, instructor)
```

---

### ✅ 3NF:

- `course_name` and `instructor` depend directly on `course_id`.
    
- No transitive dependency → **Already in 3NF.**
    

---

## ✅ **4. Orders with Transitive Dependency**

### 📌 Unnormalized Table:

```
Orders(order_id, customer_id, customer_name, customer_address, order_date)
```

### 🚫 Problem:

- `customer_name` and `customer_address` depend on `customer_id`, not `order_id` → **transitive dependency**.
    

---

### ✅ 1NF:

- Already atomic.
    

---

### ✅ 2NF:

- Primary key: `order_id`, and all attributes depend on it → **2NF**.
    

---

### ✅ 3NF:

Split transitive dependencies:

```sql
Orders(order_id, customer_id, order_date)
Customers(customer_id, customer_name, customer_address)
```

---

## ✅ **5. Sales Table with Multiple Violations**

### 📌 Unnormalized Table:

```
Sales(sale_id, customer_name, customer_email, product_ids, total_amount)
```

### 🚫 Problems:

- `product_ids` is multi-valued → violates **1NF**
    
- `customer_name` and `customer_email` depend on customer → **transitive dependency**
    

---

### ✅ 1NF:

Break product_ids into separate rows.

```sql
Sales(sale_id, customer_name, customer_email, product_id, total_amount)
```

---

### ✅ 2NF:

- Primary key: `sale_id`, `product_id`
    
- But customer data depends only on sale → **still bad**
    

---

### ✅ 3NF:

Split into:

```sql
Sales(sale_id, customer_id, product_id, total_amount)
Customers(customer_id, customer_name, customer_email)
Products(product_id, product_name)
```

