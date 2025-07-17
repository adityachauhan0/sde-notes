
---

## How to Use This Roadmap

**For each problem:**

1. **Read schema carefully** (interviewers often hide the trick in column relationships).
    
2. **Restate in your own words.**
    
3. **Sketch expected result shape** (columns, ordering, duplicates?).
    
4. **Decide operators**: filter, aggregate, join type, grouping level, window, etc.
    
5. **Write query skeleton first; fill conditions gradually.**
    
6. **Test edge cases** (ties, NULLs, duplicates, lexical vs numeric sort).
    

---

# Day 1 (Today): SQL Foundations & Filtering

Goal: Get fluent with SELECT syntax, WHERE filters, projection vs *, DISTINCT, ORDER BY, pattern matching, and basic string/len functions. Aim to finish all below today; they’re short but build muscle memory.

|Problem|Why It Matters|Hint / Thinking Process|
|---|---|---|
|**Revising the Select Query I**. ([HackerRank](https://www.hackerrank.com/challenges/revising-the-select-query/problem?utm_source=chatgpt.com "Revising the Select Query I - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/revising-the-select-query/forum?utm_source=chatgpt.com "Revising the Select Query I Discussions \| SQL - HackerRank"))|Classic filter + * vs column selection; interviews love “return all cols for condition” warm‑ups.|Filter on population > 100000 AND CountryCode='USA'. Order not required unless specified. Start with `SELECT *` and add WHERE clauses incrementally; test > vs >= carefully. ([HackerRank](https://www.hackerrank.com/challenges/revising-the-select-query/problem?utm_source=chatgpt.com "Revising the Select Query I - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/revising-the-select-query/forum?utm_source=chatgpt.com "Revising the Select Query I Discussions \| SQL - HackerRank"))|
|**Japanese Cities' Attributes**. ([HackerRank](https://www.hackerrank.com/challenges/japanese-cities-attributes/problem?utm_source=chatgpt.com "Japanese Cities' Attributes - HackerRank"), [DEV Community](https://dev.to/christianpaez/hackerrank-sql-preparation-japanese-cities-attributesmysql-12e9?utm_source=chatgpt.com "HackerRank SQL Preparation: Japanese Cities' Attributes(MySQL)"))|Reinforces full‑row retrieval + equality filter; introduces thinking about country codes as FKs.|Straight filter COUNTRYCODE='JPN'. Good chance to practice switching between `*` and named cols; think about result set size & projection minimization. ([HackerRank](https://www.hackerrank.com/challenges/japanese-cities-attributes/problem?utm_source=chatgpt.com "Japanese Cities' Attributes - HackerRank"), [DEV Community](https://dev.to/christianpaez/hackerrank-sql-preparation-japanese-cities-attributesmysql-12e9?utm_source=chatgpt.com "HackerRank SQL Preparation: Japanese Cities' Attributes(MySQL)"))|
|**Weather Observation Station 7** (pattern‑match ends‑with vowel). ([HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-7/problem?utm_source=chatgpt.com "Weather Observation Station 7 - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-7/forum?utm_source=chatgpt.com "Weather Observation Station 7 Discussions \| SQL - HackerRank"))|WHERE + string functions/regex + DISTINCT; interviewers test pattern logic frequently.|Use `LOWER(RIGHT(CITY,1)) IN (...)` or regex `[aeiou]$`; always DISTINCT to de‑dup. Watch DB‑specific case sensitivity. ([HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-7/problem?utm_source=chatgpt.com "Weather Observation Station 7 - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-7/forum?utm_source=chatgpt.com "Weather Observation Station 7 Discussions \| SQL - HackerRank"))|
|**Weather Observation Station 5** (shortest & longest city name). ([HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-5/problem?utm_source=chatgpt.com "Weather Observation Station 5 \| HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-5/forum?utm_source=chatgpt.com "Weather Observation Station 5 Discussions \| SQL - HackerRank"))|Introduces ORDER BY expressions, MIN/MAX by derived length, UNION to combine two result rows—pattern for “top N per rule.”|Compute length; order asc/desc; pick first row each side; UNION ALL; tie‑break alphabetical. Alternative: subqueries selecting MIN(LEN) & MAX(LEN). ([HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-5/problem?utm_source=chatgpt.com "Weather Observation Station 5 \| HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-5/forum?utm_source=chatgpt.com "Weather Observation Station 5 Discussions \| SQL - HackerRank"))|
|**Select ALL / Select By ID / Names variants** (quick reps—optional speed drills if time remains). See Basic Select domain list. ([HackerRank](https://www.hackerrank.com/domains/sql?utm_source=chatgpt.com "Solve SQL - HackerRank"), [Medium](https://medium.com/%40sakshisanghi0001/hackerrank-sql-solution-4731b6aae0f7?utm_source=chatgpt.com "HackerRank SQL Solution. Basic Select Queries\| -by Sakshi Jain"))|Rapid repetition cements muscle memory; also good warm‑up before live coding.|Practice writing queries _without_ looking; timebox 2 min each; vary DB flavor (MySQL vs SQL Server quoting). ([HackerRank](https://www.hackerrank.com/domains/sql?utm_source=chatgpt.com "Solve SQL - HackerRank"), [Medium](https://medium.com/%40sakshisanghi0001/hackerrank-sql-solution-4731b6aae0f7?utm_source=chatgpt.com "HackerRank SQL Solution. Basic Select Queries\| -by Sakshi Jain"))|

**Checkpoint mini‑drill:** After finishing, write from scratch: “List all cities in Japan with population > 500K ordered desc by population.” No peeking.

---

# Day 2: Aggregations & Joins

Goal: Confidently use COUNT/SUM/AVG, GROUP BY, HAVING, INNER vs LEFT JOIN, multi‑table joins, grouping by key hierarchies, de‑dup counting, and conditional aggregation.

|Problem|Why It Matters|Hint / Thinking Process|
|---|---|---|
|**Revising Aggregations – The Count Function**. ([HackerRank](https://www.hackerrank.com/challenges/revising-aggregations-the-count-function/forum?utm_source=chatgpt.com "Revising Aggregations - The Count Function Discussions \| SQL"), [Medium](https://jwjin0330.medium.com/hackerrank-sql-practice-%E2%91%A0-a5b64e87cda2?utm_source=chatgpt.com "HackerRank SQL Practice ③. Aggregation \| by jjin - Medium"))|COUNT with filter; introduces counting IDs vs rows vs DISTINCT; difference matters in interviews.|COUNT rows where population > threshold; prefer `COUNT(*)` if non‑NULL guaranteed; else choose column; consider indexing predicate cols. ([HackerRank](https://www.hackerrank.com/challenges/revising-aggregations-the-count-function/forum?utm_source=chatgpt.com "Revising Aggregations - The Count Function Discussions \| SQL"), [Medium](https://jwjin0330.medium.com/hackerrank-sql-practice-%E2%91%A0-a5b64e87cda2?utm_source=chatgpt.com "HackerRank SQL Practice ③. Aggregation \| by jjin - Medium"))|
|**Revising Aggregations – The Sum Function**. ([HackerRank](https://www.hackerrank.com/challenges/revising-aggregations-sum/problem?utm_source=chatgpt.com "Revising Aggregations - The Sum Function - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/revising-aggregations-sum/forum?utm_source=chatgpt.com "Revising Aggregations - The Sum Function Discussions - HackerRank"))|SUM with filter; numeric aggregation + WHERE vs GROUP BY single group; building block for metrics.|SUM population where District='California'; watch case sensitivity; if none match, returns NULL—wrap COALESCE if spec demands 0. ([HackerRank](https://www.hackerrank.com/challenges/revising-aggregations-sum/problem?utm_source=chatgpt.com "Revising Aggregations - The Sum Function - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/revising-aggregations-sum/forum?utm_source=chatgpt.com "Revising Aggregations - The Sum Function Discussions - HackerRank"))|
|**Population Census** (a.k.a. Asian Population). ([HackerRank](https://www.hackerrank.com/challenges/asian-population/problem?utm_source=chatgpt.com "Population Census \| HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/asian-population/forum?utm_source=chatgpt.com "Population Census Discussions \| SQL - HackerRank"))|First real JOIN + aggregate; join fact (CITY) to dimension (COUNTRY) then sum subset; common analytics pattern.|INNER JOIN CITY→COUNTRY on code; filter continent='Asia'; SUM city.population (not country pop!). Ensure qualifier to disambiguate columns. ([HackerRank](https://www.hackerrank.com/challenges/asian-population/problem?utm_source=chatgpt.com "Population Census \| HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/asian-population/forum?utm_source=chatgpt.com "Population Census Discussions \| SQL - HackerRank"))|
|**Top Competitors** (Basic Join). ([HackerRank](https://www.hackerrank.com/domains/sql/join/difficulty%3Atrue/page%3A1/?utm_source=chatgpt.com "Solve Basic Join Questions \| SQL - HackerRank"), [Medium](https://medium.com/%40bingqianwa777/hackerrank-basic-join-challenges-f11c5d2cbd73?utm_source=chatgpt.com "HackerRank basic join Challenges - Medium"))|Multi‑table joins + HAVING to filter teams scoring >0; good practice in grouping after join.|Join students, challenges, submissions; group by hacker; HAVING SUM(score)>0; ORDER BY score desc, hacker_id; read schema slowly! ([HackerRank](https://www.hackerrank.com/domains/sql/join/difficulty%3Atrue/page%3A1/?utm_source=chatgpt.com "Solve Basic Join Questions \| SQL - HackerRank"), [Medium](https://medium.com/%40bingqianwa777/hackerrank-basic-join-challenges-f11c5d2cbd73?utm_source=chatgpt.com "HackerRank basic join Challenges - Medium"))|
|**The Report** (Basic Join). ([HackerRank](https://www.hackerrank.com/domains/sql/join/difficulty%3Atrue/page%3A1/?utm_source=chatgpt.com "Solve Basic Join Questions \| SQL - HackerRank"), [Medium](https://medium.com/%40bingqianwa777/hackerrank-basic-join-challenges-f11c5d2cbd73?utm_source=chatgpt.com "HackerRank basic join Challenges - Medium"))|Conditional classification using CASE + aggregates; interviewers ask to bucket values.|Join students & grades; CASE when marks between ranges output grade + name; filter where grade≥8 show name else NULL; order grade desc then name/marks per spec. ([HackerRank](https://www.hackerrank.com/domains/sql/join/difficulty%3Atrue/page%3A1/?utm_source=chatgpt.com "Solve Basic Join Questions \| SQL - HackerRank"), [Medium](https://medium.com/%40bingqianwa777/hackerrank-basic-join-challenges-f11c5d2cbd73?utm_source=chatgpt.com "HackerRank basic join Challenges - Medium"))|
|**African Cities / Average Population of Each Continent** (grouped join practice—do 1 or both). ([HackerRank](https://www.hackerrank.com/domains/sql/join/difficulty%3Atrue/page%3A1/?utm_source=chatgpt.com "Solve Basic Join Questions \| SQL - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/asian-population/forum?utm_source=chatgpt.com "Population Census Discussions \| SQL - HackerRank"))|More join+filter reps; continent grouping.|Standard join; GROUP BY region/continent; use AVG() and CAST/ROUND as required. ([HackerRank](https://www.hackerrank.com/domains/sql/join/difficulty%3Atrue/page%3A1/?utm_source=chatgpt.com "Solve Basic Join Questions \| SQL - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/asian-population/forum?utm_source=chatgpt.com "Population Census Discussions \| SQL - HackerRank"))|

**Evening drill:** Write one query to show continent, #cities>1M, total_pop, avg_pop_of_big_cities. Explain join cardinality aloud.

---

# Day 3: Advanced Joins, Hierarchies, Dates & Interview‑Grade Design / Normalization

Goal: Tackle more complex multi‑table logic, hierarchical counting, date range grouping, correlated subqueries, and _theory questions_ on normalization (1NF→BCNF) that often show up as conceptual interview prompts.

### 3A. Advanced SQL Problem Set

|Problem|Why It Matters|Hint / Thinking Process|
|---|---|---|
|**Placements** (Advanced Join). ([HackerRank](https://www.hackerrank.com/contests/simply-sql/challenges/placements/?utm_source=chatgpt.com "Placements \| Simply SQL Question \| Contests - HackerRank"), [Medium](https://medium.com/%40kiruthickagp/hacker-rank-sql-advanced-placements-6b11d6d691e5?utm_source=chatgpt.com "Hacker rank \| SQL(Advanced) \| Placements \| by Kiruthickagp - Medium"))|Multiple tables; compare a student’s offer vs best friend’s offer; introduces self‑reference join pattern.|Join Students→Friends (self relationship) then join both to Packages twice (student & friend); WHERE friend_salary > student_salary; ORDER BY friend_salary; watch aliasing! ([HackerRank](https://www.hackerrank.com/contests/simply-sql/challenges/placements/?utm_source=chatgpt.com "Placements \| Simply SQL Question \| Contests - HackerRank"), [Medium](https://medium.com/%40kiruthickagp/hacker-rank-sql-advanced-placements-6b11d6d691e5?utm_source=chatgpt.com "Hacker rank \| SQL(Advanced) \| Placements \| by Kiruthickagp - Medium"))|
|**New Companies** (Hierarchy rollups). ([HackerRank](https://www.hackerrank.com/challenges/the-company/problem?utm_source=chatgpt.com "New Companies \| HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/the-company/forum?utm_source=chatgpt.com "New Companies Discussions \| SQL - HackerRank"))|Aggregating counts across hierarchical levels (lead mgr → senior mgr → mgr → employee); interviews love org charts.|COUNT DISTINCT codes from each table per company_code; LEFT JOIN chain or aggregate each subtable then join; ORDER BY company_code lexicographically (string sort). ([HackerRank](https://www.hackerrank.com/challenges/the-company/problem?utm_source=chatgpt.com "New Companies \| HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/the-company/forum?utm_source=chatgpt.com "New Companies Discussions \| SQL - HackerRank"))|
|**SQL Project Planning** (Date range grouping). ([HackerRank](https://www.hackerrank.com/challenges/sql-projects/problem?utm_source=chatgpt.com "SQL Project Planning \| HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/sql-projects/forum?utm_source=chatgpt.com "SQL Project Planning Discussions - HackerRank"))|Consecutive date bucketing—tests window functions, LAG/ROW_NUMBER deltas, or self‑join gaps‑and‑islands technique.|Treat each task day as row; “islands” form when start_date not one day after prior end_date; group islands with (start_date - ROW_NUMBER()) trick; then MIN/ MAX per island; order by duration then start. ([HackerRank](https://www.hackerrank.com/challenges/sql-projects/problem?utm_source=chatgpt.com "SQL Project Planning \| HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/sql-projects/forum?utm_source=chatgpt.com "SQL Project Planning Discussions - HackerRank"))|
|**Ollivander's Inventory / Challenges** (from Basic Join list—pick one for extra join practice). ([HackerRank](https://www.hackerrank.com/domains/sql/join/difficulty%3Atrue/page%3A1/?utm_source=chatgpt.com "Solve Basic Join Questions \| SQL - HackerRank"), [Medium](https://medium.com/%40bingqianwa777/hackerrank-basic-join-challenges-f11c5d2cbd73?utm_source=chatgpt.com "HackerRank basic join Challenges - Medium"))|Forces multi‑condition join + MIN to pick cheapest wand meeting spec; common “top‑N per group” pattern.|Partition by wand_id & power; pick min_coins_needed; join on age restriction; filter duplicates via subquery or window. ([HackerRank](https://www.hackerrank.com/domains/sql/join/difficulty%3Atrue/page%3A1/?utm_source=chatgpt.com "Solve Basic Join Questions \| SQL - HackerRank"), [Medium](https://medium.com/%40bingqianwa777/hackerrank-basic-join-challenges-f11c5d2cbd73?utm_source=chatgpt.com "HackerRank basic join Challenges - Medium"))|

### 3B. Normalization Theory Drills (High‑Yield Interview Talking Points)

Work these _concept_ problems; they’re short numeric/MCQ style but push you to reason about functional dependencies (FDs), partial vs transitive dependency, and BCNF. After each, articulate: **keys, dependencies, violation, fix.**

|Problem|Normal Form Concept|Hint / What to Practice|
|---|---|---|
|**Database Normalization #1 – 1NF**. ([HackerRank](https://www.hackerrank.com/blog/database-interview-questions-you-should-know/?utm_source=chatgpt.com "15 Database Interview Questions You Should Know - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-1-1nf/forum?utm_source=chatgpt.com "Database Normalization #1 - 1NF Discussions - HackerRank"))|Atomicity; remove repeating groups / multi‑valued attributes.|Split multicolor column into child table with (ProductID, Color); prices stay in product table; count resulting rows/cols per prompt. Practice spotting repeating lists. ([HackerRank](https://www.hackerrank.com/blog/database-interview-questions-you-should-know/?utm_source=chatgpt.com "15 Database Interview Questions You Should Know - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-1-1nf/forum?utm_source=chatgpt.com "Database Normalization #1 - 1NF Discussions - HackerRank"))|
|**Database Normalization #2 – 1/2/3 NF**. ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-123nf/problem?utm_source=chatgpt.com "Database Normalization #2 - 1/2/3 NF - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-123nf/forum/comments/512166?utm_source=chatgpt.com "Database Normalization #2 - 1/2/3 NF Discussions - HackerRank"))|Partial vs transitive dependency; mapping cardinality clues which NF is violated.|Many Zip→City hints at non‑key dependency; ask: is (PK, City, Zip)? If Zip determines City, City is transitively dependent; thus not 3NF; check if partial vs composite key. ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-123nf/problem?utm_source=chatgpt.com "Database Normalization #2 - 1/2/3 NF - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-123nf/forum/comments/512166?utm_source=chatgpt.com "Database Normalization #2 - 1/2/3 NF Discussions - HackerRank"))|
|**Database Normalization #3** (student↔course many‑to‑many). ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-3/problem?utm_source=chatgpt.com "Database Normalization #3 - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-3/forum/comments/113813?utm_source=chatgpt.com "Database Normalization #3 Discussions - HackerRank"))|Resolving M:N relationships via junction table; avoids repeating groups / update anomalies.|Create enrollment table (student_id, course_id, ...). Recognize that storing arrays violates 1NF & leads to anomalies; decomposition to 3NF. ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-3/problem?utm_source=chatgpt.com "Database Normalization #3 - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-3/forum/comments/113813?utm_source=chatgpt.com "Database Normalization #3 Discussions - HackerRank"))|
|**Database Normalization #4** (2NF→3NF table counts). ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-4/problem?utm_source=chatgpt.com "Database Normalization #4 - HackerRank"), [HackerRank](https://www.hackerrank.com/blog/database-interview-questions-you-should-know/?utm_source=chatgpt.com "15 Database Interview Questions You Should Know - HackerRank"))|Understand attribute promotion when moving to higher NF; reasoning about minimum decomposition.|If each 2NF table has key + non‑key, to reach 3NF you combine those sharing same key dependencies; compute grouping by same determinant. ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-4/problem?utm_source=chatgpt.com "Database Normalization #4 - HackerRank"), [HackerRank](https://www.hackerrank.com/blog/database-interview-questions-you-should-know/?utm_source=chatgpt.com "15 Database Interview Questions You Should Know - HackerRank"))|
|**Database Normalization #5** (FD → up to BCNF). ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-5/forum?utm_source=chatgpt.com "Database Normalization #5 Discussions - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-5/forum/comments/165264?utm_source=chatgpt.com "Database Normalization #5 Discussions - HackerRank"))|Detect when determinant not a (candidate) key—BCNF violation; subtle difference from 3NF.|List all FDs; compute closures; if any determinant not candidate key, BCNF fails; else passes; answer in {1,2,3,3.5}. ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-5/forum?utm_source=chatgpt.com "Database Normalization #5 Discussions - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-5/forum/comments/165264?utm_source=chatgpt.com "Database Normalization #5 Discussions - HackerRank"))|
|**Database Normalization #6** (max NF from determinants). ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-6/problem?utm_source=chatgpt.com "Database Normalization #6 - HackerRank"), [Database Administrators Stack Exchange](https://dba.stackexchange.com/questions/218325/problem-in-testing-normal-form?utm_source=chatgpt.com "Problem in testing normal form"))|Mixed dependencies; test your ability to evaluate highest satisfied NF.|For each FD set, check 1NF (atomic), 2NF (no partial), 3NF (no transitive non‑prime → prime), BCNF (every determinant candidate key). Pick highest all satisfy. ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-6/problem?utm_source=chatgpt.com "Database Normalization #6 - HackerRank"), [Database Administrators Stack Exchange](https://dba.stackexchange.com/questions/218325/problem-in-testing-normal-form?utm_source=chatgpt.com "Problem in testing normal form"))|

---

## Suggested 3‑Day Study Schedule (adjustable)

**Day 1 (Jul 17):** Do Foundations set (5 core + optional drills). Target 90 mins. Then read 1NF/2NF quick notes (#1,#2) before bed (20 mins).  
**Day 2 (Jul 18):** Morning: Aggregations (first 3). Afternoon: Basic Join set (next 3). Evening: speak aloud explanations (record yourself).  
**Day 3 (Jul 19):** Advanced set (Placements, New Companies, Project Planning, Ollivander's/Challenges). Finish Normalization #3‑#6. Do one full mock: interviewer asks you to design simple Student‑Course‑Enrollment schema and write 3 queries; practice whiteboard.  
**Interview Eve Quick Review (night of Jul 19/early Jul 20):** Flashcards: definitions of 1NF‑BCNF; difference between INNER/LEFT; GROUP BY vs WHERE; window vs aggregate; indexing tip (create index on join key + filter).

---

## Core DBMS Concept Flashcards (Make & Review)

Create short cards for:

- **Primary vs Candidate vs Foreign Key** (ties to join correctness in HackerRank joins). ([HackerRank](https://www.hackerrank.com/challenges/the-company/problem?utm_source=chatgpt.com "New Companies | HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/the-company/forum?utm_source=chatgpt.com "New Companies Discussions | SQL - HackerRank"))
    
- **Partial vs Transitive Dependency** (why 2NF/3NF). ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-123nf/problem?utm_source=chatgpt.com "Database Normalization #2 - 1/2/3 NF - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-123nf/forum/comments/512166?utm_source=chatgpt.com "Database Normalization #2 - 1/2/3 NF Discussions - HackerRank"))
    
- **BCNF stricter than 3NF** (determinant must be candidate key). ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-5/forum?utm_source=chatgpt.com "Database Normalization #5 Discussions - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-5/forum/comments/165264?utm_source=chatgpt.com "Database Normalization #5 Discussions - HackerRank"))
    
- **Decomposing M:N** (student/course). ([HackerRank](https://www.hackerrank.com/challenges/database-normalization-3/problem?utm_source=chatgpt.com "Database Normalization #3 - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-3/forum/comments/113813?utm_source=chatgpt.com "Database Normalization #3 Discussions - HackerRank"))
    

---

## Rapid Practice Strategy

1. **Timebox**: 10 min / easy; 20 min / medium. If stuck, peek only at schema, not solution.
    
2. **Verbalize** constraints: “Need unique row per company_code; duplicates exist; use DISTINCT or aggregate.”
    
3. **Alternate DB engines**: Try MySQL then re‑write in SQL Server (TOP vs LIMIT), improves fluency. Many discussion threads show engine differences—scan them after you’ve attempted. ([HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-7/forum?utm_source=chatgpt.com "Weather Observation Station 7 Discussions | SQL - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-5/forum?utm_source=chatgpt.com "Weather Observation Station 5 Discussions | SQL - HackerRank"))
    
4. **Check sort semantics**: String vs numeric sort (e.g., company_code like C_1, C_10). ([HackerRank](https://www.hackerrank.com/challenges/the-company/problem?utm_source=chatgpt.com "New Companies | HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/the-company/forum?utm_source=chatgpt.com "New Companies Discussions | SQL - HackerRank"))
    

---

## Want More / Stretch (if time allows)

- Work a few **Weather Observation Station** pattern problems (#8 vowels both ends; #9 not starting with vowel) to sharpen LIKE/REGEXP logic. ([Medium](https://gracejeung.medium.com/weather-observation-station-6857a3691bae?utm_source=chatgpt.com "HackerRank SQL | Weather Observation Station - Grace Jeung"), [HackerRank](https://www.hackerrank.com/challenges/weather-observation-station-7/problem?utm_source=chatgpt.com "Weather Observation Station 7 - HackerRank"))
    
- Read HackerRank blog **“15 Database Interview Questions You Should Know”** for conceptual refreshers beyond coding (indexes, ACID, transactions). ([HackerRank](https://www.hackerrank.com/blog/database-interview-questions-you-should-know/?utm_source=chatgpt.com "15 Database Interview Questions You Should Know - HackerRank"))
    

---

### Quick Read‑Before‑Interview Normalization Script (practice saying out loud)

> **1NF:** No repeating groups; atomic values.  
> **2NF:** 1NF + every non‑key fully dependent on _whole_ composite key (no partial).  
> **3NF:** 2NF + no transitive dependency of non‑key on key via another non‑key.  
> **BCNF:** For every FD X→Y, X is a candidate key (stronger than 3NF).  
> Rehearse with Product‑Color example (#1), Zip/City (#2), Student‑Course (#3), and BCNF FD set (#5). ([HackerRank](https://www.hackerrank.com/blog/database-interview-questions-you-should-know/?utm_source=chatgpt.com "15 Database Interview Questions You Should Know - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-123nf/problem?utm_source=chatgpt.com "Database Normalization #2 - 1/2/3 NF - HackerRank"), [HackerRank](https://www.hackerrank.com/challenges/database-normalization-5/forum?utm_source=chatgpt.com "Database Normalization #5 Discussions - HackerRank"))

---
