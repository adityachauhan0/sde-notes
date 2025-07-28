# Design Leetcode
## 0. Problem Statement

Design an online coding platform (like LeetCode): users solve problems in many languages, code runs securely, results & stats are recorded, and contests are hosted.

---

## 1. Requirements

### Functional

- Browse/search problems by tag/difficulty.
    
- Code editor: **Run (samples)** vs **Submit (hidden tests)**.
    
- Auto-judge (AC/WA/TLE/MLE/RE).
    
- User auth/profile/stats/streaks.
    
- Contests: register, timed submissions, live leaderboard.
    
- Admin CRUD for problems and testcases.
    
- Rate limiting for abuse prevention.
    

### Non‑Functional

- P95 read latency <150 ms globally; P95 judge latency <5 s (easy) / <15 s (hard).
    
- 99.9% availability.
    
- Secure isolation of untrusted code.
    
- Elastic compute for bursty submissions.
    
- Strong consistency for submission status; eventual for analytics.
    

---

## 2. Rough Sizing (sample assumptions)

- 15 M MAU / 1.5 M DAU / 80k peak concurrent.
    
- Read QPS ~60k; submission QPS peak ~3k.
    
- 500 M+ historical submissions (2–3 KB metadata each, code ~5–20 KB).
    
- Problems: 3k–10k; test bundles ~50 MB/problem (store in object storage).
    

---

## 3. High-Level Architecture

```
Users (Web/Mobile/CLI)
        |
        v
+--------------------+
| CDN / Edge Cache   |
+--------------------+
        |
        v
+--------------------+        +-----------------+
| API Gateway        |<------>| Auth Service    |
+--------------------+        +-----------------+
   |        |    \
   |        |     \                         +------------------+
   |        |      \-----> Search Service ->| OpenSearch/ES     |
   |        |                                +------------------+
   |        |
   |   +------------------+     +------------------+
   |   | Problem Service  |---->| Problem DB (SQL) |
   |   +------------------+     +------------------+
   |
   |   +------------------+     +---------------------+
   +-->| Submission Svc   |---->| Submission DB (SQL) |
       +------------------+     +---------------------+
                 |
                 v
          +----------------+
          |  Job Queue     |  (Kafka/SQS/PubSub)
          +----------------+
                 |
                 v
       +-----------------------+
       | Judge Orchestrator    |
       +-----------------------+
          /        |        \
         /         |         \
+--------------+ +--------------+ +--------------+
| Worker Pool  | | Worker Pool  | | Worker Pool  |  (multi-region)
| (Firecracker)| | (Docker)     | | (GPU, etc.)  |
+--------------+ +--------------+ +--------------+
        |               |               |
        v               v               v
  Testcases / Code / Logs in Object Storage (S3/GCS/MinIO)
        |
        v
+------------------+
| Result Writer    |
+------------------+
        |
        v
 Cache (Redis)  <---->  Leaderboards / Stats Service  <---->  Analytics DB/WH (ClickHouse/BigQuery)
```

---

## 4. Critical Flow (Submission → Judge → Result)

```
Client
  | POST /submissions (code, lang, RUN/SUBMIT)
  v
Submission Service
  |-- persist metadata (SQL)
  |-- store code blob (S3)
  |-- enqueue job (Kafka/SQS)
  v
Judge Orchestrator
  |-- lease job
  |-- pick idle worker (Redis set / gRPC)
  v
Worker
  |-- fetch test bundle (S3)
  |-- compile/run in microVM/container (timeouts, mem limits)
  |-- compute verdict, capture logs
  v
Result Writer
  |-- update Submission DB
  |-- publish event (WebSocket topic)
  |-- invalidate caches / update stats
  v
Client gets verdict via poll or push
```

ASCII sequence:

```
User -> API -> SubmissionSvc -> DB
                   |              \
                   |               -> Queue
                   v
              Orchestrator -> Worker -> S3 -> Worker -> Result
                                          ^              |
                                          |--------------|
Result -> SubmissionSvc -> DB -> WebSocket -> User
```

---

## 5. Data Model (simplified ER ASCII)

```
+-------+        +-------------+        +-----------+
| User  |1----*< | Submission  |> *----1| Problem   |
+-------+        +-------------+        +-----------+
| id    |        | id          |        | id        |
| name  |        | user_id     |        | title     |
| ...   |        | problem_id  |        | tags[]    |
+-------+        | lang        |        | ...       |
                 | verdict     |        +-----------+
                 | runtime_ms  |
                 | created_at  |        +--------------+
                 +-------------+   1--*<| TestBundle   |
                                      | +--------------+
                                      | | problem_id   |
                                      | | version      |
                                      | | s3_path      |
                                      | | type         |
                                      | +--------------+

+----------+       +-----------------+
| Contest  |1---*< | ContestScoreRow |
+----------+       +-----------------+
| id       |       | contest_id      |
| start_ts |       | user_id         |
| end_ts   |       | score           |
| problems[]       | penalty         |
+----------+       +-----------------+
```

---

## 6. Judge/Sandbox Design

```
+---------------------+
| Worker Node         |
|  +---------------+  |
|  | VM Snapshot   |<-+-- Pre-baked per language (fast boot)
|  +---------------+  |
|  | Firecracker VM|  |
|  | (seccomp,     |  |
|  | cgroups, no   |  |
|  | net)          |  |
|  +---------------+  |
|  | Runner Agent  |-- compile/run, parse result
|  +---------------+  |
+---------------------+
```

Key points: microVM isolation, read-only FS, throttle CPU/mem, kill after timeout.

---

## 7. Storage & Caching

- **SQL (Aurora/Postgres/MySQL):** Users, problems meta, fresh submissions. Partition submissions (by user_id hash or time). Read replicas for scale.
    
- **Object Storage:** Testcases, code blobs, logs, editorials. Versioned & replicated.
    
- **Redis:**
    
    - Hot problem meta (`problem:{id}:v{n}`)
        
    - User stats/streak counters
        
    - Contest leaderboard top-K
        
    - Rate-limits (token bucket)
        
- **Search (OpenSearch/Elastic):** Full-text search on titles/editorials.
    
- **Cold Analytics (ClickHouse/BigQuery/S3 Parquet):** Historical queries, BI, ML features.
    

---

## 8. API Sketch

```
GET  /problems?tag=&difficulty=&search=
GET  /problems/{slug}
POST /submissions {problemId, lang, code, mode:RUN|SUBMIT}
GET  /submissions/{id}
WS   /live/submissions/{id}

POST /contests/{id}/register
GET  /contests/{id}/leaderboard
GET  /users/{id}/stats

Admin:
POST /problems
POST /problems/{id}/tests
```

---

## 9. Scaling/HA Strategies

- **Stateless services** behind k8s + HPA.
    
- **Judge workers** auto-scale on queue lag / in-flight jobs; use spot/cheap instances.
    
- **DB scaling:** Read replicas; partition submissions, or hybrid SQL + NoSQL.
    
- **Queue:** Kafka (topic per priority) or SQS; DLQ for poison jobs.
    
- **Multi-region:**
    
    - Edge CDN for problem pages.
        
    - Regional judge clusters.
        
    - Global object storage replication.
        
    - Use global load balancer; sticky region for sessions.
        

---

## 10. Consistency & Reliability

- **Idempotent submissions:** Client gets `submission_id`; retransmit safe.
    
- **Exactly-once-ish judge:** Orchestrator uses leases; worker ACK writes result via transactional outbox.
    
- **Stats eventual:** Update counters async; recompute nightly for correctness.
    
- **Failures:**
    
    - Worker crash → lease expires → requeue.
        
    - Queue outage → buffer in DB table.
        
    - DB failover → driver retries, circuit breaker.
        

---

## 11. Security

- WAF + rate limits.
    
- JWT/OAuth2 auth; bcrypt/argon2 hashing.
    
- Encrypt at rest (KMS) & TLS everywhere.
    
- Sandbox: syscall filtering, no outbound net, tmpfs, cgroups.
    
- Audit logs for admin ops; GDPR deletion workflow.
    

---

## 12. Observability

- Metrics: queue depth, judge P95, compile failure rate, cache hit %, DB slow queries.
    
- Tracing: API→orchestrator→worker (Jaeger/OTel).
    
- Logs: structured JSON, centralized (ELK/Loki).
    
- Alerting: SLO-based alerts, anomaly detection on submission spikes.
    

---

## 13. Contest/Leaderboard Pipeline

```
[submission_result topic]
          |
          v
   Contest Service
     |-- validate contest window
     |-- compute score delta (penalty rules)
     |-- update in-memory leaderboard (Redis sorted sets)
     |-- batch persist (SQL)
          |
          v
   /leaderboard endpoint (read from Redis)
```

Freeze after contest end to SQL, publish static snapshot.

---

## 14. Trade-offs & Alternatives

|Decision|Chosen|Alternative|Why|
|---|---|---|---|
|Isolation|Firecracker|Docker only|Better security; small boot penalty|
|Submissions store|SQL partitioned|NoSQL (Cassandra)|SQL simplifies joins, transactions|
|Result delivery|Poll + WS|Long-poll only|WS improves UX; keep polling fallback|
|Global queue|Per-region queues|Global single Kafka|Regional queues reduce latency/BLAST radius|
|Search|OpenSearch|DB full-text|Scalability, relevance tuning|

---

## 15. Future Enhancements

- ML-driven recommendations & difficulty personalization.
    
- Collaborative rooms / mock interviews.
    
- IDE plugins (VSCode) using same submission API.
    
- AI hints/explanations after WA.
    
- Company portals & custom problem sets.
    

---

## 16. What to Deep-Dive Next?

- Judge Orchestrator algorithms (load balancing, lease protocol).
    
- Problem/testcase versioning & cache invalidation.
    
- Data partitioning scheme for submissions.
    
- Real-time leaderboard consistency model.
    
- Cost optimization (spot instances, cold storage lifecycle).
    

---


# System Design Interview: **Design a URL Shortener (e.g., TinyURL/bit.ly)**

---

## 0. Problem Statement (30-sec pitch)

Build a service that converts long URLs into short codes (e.g., `https://bit.ly/xYz123`). When users hit the short URL, they get redirected to the original long URL. Track basic stats (clicks, creation time).

---

## 1. Requirements

### Functional

- Create short URL from a long URL.
    
- Redirect short URL → original URL.
    
- Optional: custom alias (if available), link expiration, basic analytics (click count, last access time).
    
- Admin: delete/disable malicious links.
    

### Non-Functional

- Low latency for redirect (<50 ms P95).
    
- High availability (people share links everywhere).
    
- Handle read-heavy traffic (redirects) vs fewer writes (shorten operations).
    
- Prevent collisions & brute-force scanning.
    
- Scale to billions of URLs.
    

---

## 2. Back-of-the-Envelope Numbers (Assume)

- 100M new URLs/year ≈ ~3 URLs/sec average, peaks 10x.
    
- Redirects 1000x more than creations → ~3K QPS average, peaks higher.
    
- Each record: longURL (~500 bytes avg), metadata (~200 bytes). ~1 KB/record → 100M KB ≈ 100 GB/year (metadata only; easy to store).
    
- Link lifetime: “forever”—design for pruning/archiving later.
    

---

## 3. Core API Sketch

```http
POST /api/v1/shorten
Body: { "longUrl": "...", "customAlias": "myLink", "expireAt": "2025-12-31T00:00:00Z" }
Resp: { "shortUrl": "https://sho.rt/AbC123" }

GET /AbC123        -> 301 Redirect to original URL

GET /api/v1/stats/AbC123
Resp: { "createdAt": "...", "redirectCount": 12345, "lastAccessAt": "..." }
```

---

## 4. Data Model (Simplified)

```
URLMapping
-----------
short_code (PK)   VARCHAR(8)   // e.g., "AbC123"
long_url          TEXT
created_at        TIMESTAMP
expire_at         TIMESTAMP NULL
click_count       BIGINT
creator_user_id   BIGINT NULL
is_active         BOOL
```

Indexes:

- PK on `short_code`
    
- Secondary index on `creator_user_id` (to list user’s links)
    
- Optional TTL index if using a NoSQL with TTL.
    

---

## 5. High-Level Architecture (ASCII)

```
Clients (Web / Mobile / API)
          |
          v
+-------------------+
|  CDN / Edge Cache |  <- cache popular redirects
+-------------------+
          |
          v
+-------------------+         +--------------------+
|  API Gateway      | <-----> | Auth / RateLimit   |
+-------------------+         +--------------------+
     |         \
     |          \  POST /shorten
     |           \
     v            v
+---------------------+     +----------------------+
| URL Service         |     | Stats/Analytics Svc  |
+---------------------+     +----------------------+
     |     \                       ^
     |      \                      |
     v       v                     |
+-----------+  +----------------+  |
| KV Store  |  | SQL/NoSQL DB   |  |
| (Redis)   |  | (Cassandra/My) |  |
+-----------+  +----------------+  |
     ^                                \
     |  GET /{code}                    \ batch/stream
     |                                  v
     +---------------------------+  +---------------------+
                                 |  | Stream (Kafka/KDS) |
                                 +->| Click Aggregator   |
                                    +---------------------+
```

---

## 6. How to Generate the Short Code?

**Goal:** Unique, short, URL-safe string.

### Options

1. **Hash + Collision Handling**
    
    - Take MD5/SHA1 of long URL → base62 encode first N chars.
        
    - If collision → append random chars or rehash with salt.
        
2. **Auto-Increment ID + Base62**
    
    - Maintain a counter (e.g., in DB or Redis).
        
    - Convert integer to base62 (0-9, a-z, A-Z) → short code.
        
    - Easy, ordered, no collisions. Need to avoid single counter bottleneck.
        
3. **Snowflake / UUID-based + Base62**
    
    - Use distributed ID generators.
        

**Interview Tip:** Say you’ll start with **auto-increment + base62** (simple) and mention sharding counter or using Snowflake if needed.

---

## 7. End-to-End Flows

### Create Short URL

```
Client -> API /shorten
   |
   v
1. Validate URL
2. Generate code (Base62(counter))
3. Store {code → longURL, meta} in DB
4. Return short URL
```

### Redirect

```
User hits https://sho.rt/AbC123
  |
  v
1. CDN/Edge checks cache
2. If miss -> API/Redirect service
3. Read longURL from cache (Redis) else DB
4. 301 redirect to longURL
5. Async record click event (fire-and-forget to queue)
```

**ASCII Sequence:**

```
Client --> CDN --> API --> Cache? --> DB
                     |       |         |
                     |<------|         |
                     | Redirect 301    |
                     v
              Analytics Queue --> Stats Processor --> DB
```

---

## 8. Caching Strategy

- **Edge/CDN**: Cache popular redirects; set short TTL to allow disabling malicious links.
    
- **Redis**: `GET /{code}` → fetch from Redis first. Cache aside pattern.
    
- Invalidate cache if link disabled/expired.
    

---

## 9. DB Choices & Scaling

- **Writes (shorten)**: comparatively low → SQL fine. Partition on `short_code` or just single instance with replicas.
    
- **Reads (redirects)**: heavy, but read-mostly → shard by `short_code` hash if using NoSQL (Cassandra/DynamoDB).
    
- **Hybrid**: SQL for metadata + Redis for hot path.
    

---

## 10. Availability & Consistency

- **Redirect path:** prioritize availability (serve from cache).
    
- **Creation path:** strong consistency for uniqueness (transaction on code).
    
- **Stats:** eventual consistency (aggregate clicks asynchronously).
    

CAP thinking: For redirect lookups, choose AP (eventual) because stale read (e.g., one more click) is okay.

---

## 11. Security & Abuse

- Validate user-submitted URLs (no XSS/data: URIs).
    
- Rate limit shorten API (prevent spam).
    
- Blacklist malicious domains.
    
- Add CAPTCHA for anonymous users.
    

---

## 12. Observability

- Metrics: QPS shorten/redirect, latency, cache hit %, error rates.
    
- Logs: per redirect (sampled) + full for creation.
    
- Alerts when sudden spike on a link (possible attack).
    

---

## 13. Enhancements / Extensions

- Custom domains for enterprise (e.g., go.company.com).
    
- QR code generation.
    
- Per-country/device analytics.
    
- Link preview cards.
    
- Expiring or one-time-use links.
    
- A/B testing short code lengths.
    

---

## 14. Trade-offs Cheat Sheet

|Issue|Option A|Option B|Pick & Why|
|---|---|---|---|
|Code gen|Counter+Base62|Hash+collision|Counter is simple & deterministic|
|Store|SQL|NoSQL KV|SQL OK early; migrate to NoSQL as scale grows|
|Click stats|Sync update|Async stream|Async keeps redirect path fast|
|Cache invalidation|TTL|Explicit delete|TTL simple; plus admin delete call|

---

## 15. How to Present in Interview (Timeline)

1. **Clarify scope** (30–60s).
    
2. **Reqs & traffic** (2 min).
    
3. **API + data model** (3 min).
    
4. **High-level design & flows** (5–7 min).
    
5. **Deep dive** (ID gen, cache, DB scaling) (5–7 min).
    
6. **Discuss trade-offs & extensions** (2–3 min).
    
7. **Wrap-up & questions** (1 min).
    

---

# System Design Interview: **Design Instagram (Image Feed Sharing Platform)**

---

## 0. Problem Statement (Elevator Pitch)

Build a platform where users post photos/videos, follow others, and see a personalized, reverse‑chronological (or ranked) feed. Support likes/comments, notifications, and basic search.

---

## 1. Requirements

### Functional

- User signup/login, follow/unfollow.
    
- Post photo/video with caption & tags.
    
- Generate a **home feed** (timeline of followed users’ posts).
    
- Like/comment on posts.
    
- Notifications (someone liked/commented/followed).
    
- Basic search: users, hashtags.
    
- Optional: Stories, reels, messaging (out of scope here).
    

### Non-Functional

- Low feed latency (<200 ms P95 read).
    
- High availability (99.9%).
    
- Write-heavy on media upload; read-heavy on feeds.
    
- Support millions of DAU; peak traffic spikes (celebrity posts).
    
- Strong consistency for actions like follow/like; eventual OK for counters.
    

Assumptions (numbers help sizing discussions):

- 50M MAU, 5M DAU, 500k peak concurrent.
    
- 2M new posts/day (~23/sec avg; 10x spikes).
    
- Average photo 500KB (compressed), video ~5–20MB (we’ll mostly focus on photos).
    
- Each user follows ~200 accounts on average.
    

---

## 2. High-Level Architecture (ASCII)

```
Clients (iOS/Android/Web)
        |
        v
+------------------+
| CDN (images/vid) |
+------------------+
        |
        v
+--------------------+       +---------------------+
| API Gateway        |<----->| Auth/Rate Limit Svc |
+--------------------+       +---------------------+
   |        |     \
   |        |      \-------------------+
   |        |                          \
   v        v                           v
+-----------+-----------+      +---------------------+
| User/Follow Service   |      | Post Service        |
+-----------+-----------+      +---------------------+
            |                          |
            v                          v
      User DB (SQL)             Post DB (SQL/NoSQL)
            |                          |
            +------------+-------------+
                         |
                         v
                 Feed Generation Layer
            (Fan-out on write/read, Redis)
                         |
                         v
                   Feed Cache Store
                 (Redis / Cassandra)
                         |
                         v
                +------------------+
                | Notification Svc |
                +------------------+
                         |
                         v
                   Push/Email/WebSocket

Media Path:
Client -> Upload Svc -> Object Storage (S3/GCS) -> CDN edge delivery
```

---

## 3. Core Data Models (Simplified)

```
User(id, username, name, bio, created_at, ...)
Follow(follower_id, followee_id, created_at)           // directed edge

Post(id, user_id, media_url, caption, tags[], created_at)

FeedItem(user_id, post_id, created_at)                  // materialized feed rows

Like(post_id, user_id, created_at)
Comment(id, post_id, user_id, text, created_at)

Notification(id, to_user_id, type, actor_id, post_id, created_at, seen)
```

Indexes:

- Follow: idx on follower_id (to get my followees quickly).
    
- Post: idx on user_id, created_at (fetch user’s posts).
    
- FeedItem: composite (user_id, created_at DESC).
    
- Like/Comment: post_id idx.
    

---

## 4. APIs (Representative)

```http
POST /users          (signup)
POST /login
POST /follow         {targetUserId}
DELETE /follow/{id}

POST /posts          (upload metadata, get upload URL)
GET  /posts/{id}
GET  /users/{id}/posts?limit=...

GET  /feed?cursor=...
POST /posts/{id}/like
POST /posts/{id}/comment  {text}

GET  /notifications?cursor=...
```

Internal:

- FeedService: `PUT /fanout/{post_id}` (distribute to followers)
    
- NotificationService subscribes to events (like, comment, follow).
    

---

## 5. Media Upload & Delivery Flow

1. Client asks for upload URL → Upload Service issues pre-signed S3 URL.
    
2. Client uploads media directly to S3 (bypasses API servers).
    
3. Post metadata saved in DB (media_url = S3 path).
    
4. CDN fronts S3 for fast global delivery.
    

```
Client -> API (request upload URL)
Client -> S3 (PUT media)
API -> Post Svc (save post record)
CDN -> end users fetch media
```

---

## 6. Feed Generation Strategies

**Two classic models:**

1. **Fan-out on Write** (Push):
    
    - When user A posts, push that post_id into all followers’ feed lists.
        
    - Pros: fast read; Cons: heavy write for celebrities (millions of followers).
        
2. **Fan-out on Read** (Pull):
    
    - On feed request, dynamically gather posts from followees, sort & return.
        
    - Pros: cheap write; Cons: slow read for users with many followees.
        

**Hybrid (common realistic choice):**

- Push for “normal” users (<100k followers).
    
- Pull for “celeb” users (store their posts separately; followers merge on read).
    

Implementation detail:

- Use Redis lists or Cassandra wide rows keyed by `user_id` to store feed items.
    
- Use background workers to fan-out posts.
    

ASCII for fan-out on write:

```
New Post -> Post Svc -> Fetch follower IDs (Follow DB/Cache)
                   \
                    -> For each follower:
                         enqueue "Insert FeedItem(follower, post)"
                           |
                           v
                      Feed Store (Redis/Cassandra)
```

Feed read path:

```
Client -> API /feed?cursor=... 
  |
  v
Feed Svc -> Redis/Cassandra (get latest N post_ids)
          -> Batch query Post DB/Cache for post details
          -> Return to client
```

---

## 7. Caching Strategy

- **Redis** for:
    
    - User sessions, rate limits.
        
    - Feed lists per user (latest 500 post_ids).
        
    - Post metadata hot cache.
        
- **CDN edge** for media files.
    
- **Write-behind/Write-through** patterns for counters (likes) to reduce DB writes.
    

---

## 8. Scaling & Partitioning

- **User/Follow DB**: Relational (Postgres/MySQL) with sharding by user_id hash when needed; or migrate to Cassandra for Follow edges.
    
- **Post storage**: Could be in Cassandra (wide rows by user_id) for high write throughput.
    
- **Feed store**: Redis + Cassandra. Partition by user_id.
    
- **Microservices**: Deploy behind Kubernetes; use autoscaling on CPU/QPS.
    

Queue/Stream (Kafka/Pulsar) for:

- Post fan-out tasks.
    
- Notification events.
    
- Analytics pipelines.
    

---

## 9. Consistency & Reliability

- **Strong**: follow relationship, like toggling (use transactions or idempotent ops).
    
- **Eventual**: feed materialization, like counters, notifications read status replication.
    

Failure handling:

- If fan-out workers fail, jobs remain in queue (retry).
    
- Redis outage → fallback to pull model temporarily.
    
- CDN outage → backup domain or multi-CDN.
    

---

## 10. Notifications

- Subscribe to events (post, like, comment, follow) via Kafka.
    
- Notification service creates records and pushes via:
    
    - Mobile push (FCM/APNs)
        
    - WebSocket/long-poll
        
    - Email (optional)
        

Store notifications in SQL/NoSQL keyed by `to_user_id` with pagination.

---

## 11. Security & Privacy

- OAuth/JWT tokens for API calls.
    
- Media URLs signed or public? Usually public behind unguessable paths + CDN.
    
- Rate limiting & spam detection for comments/likes.
    
- GDPR: delete user → remove posts or anonymize.
    

---

## 12. Observability

- Metrics: feed latency, fan-out queue lag, cache hit rate, media upload errors.
    
- Tracing: API → service → DB (OpenTelemetry/Jaeger).
    
- Logs: aggregated in ELK / Loki.
    
- Alerts: SLO breaches (P95 > target), queue backlogs, error spikes.
    

---

## 13. Trade-offs (Quick Table)

|Problem|Option A (Chosen)|Option B|Why|
|---|---|---|---|
|Feed gen|Hybrid push/pull|Pure push or pure pull|Balance cost & latency|
|Store posts|Cassandra/SQL + Redis cache|Only SQL|Better write scaling|
|Media hosting|S3 + CDN|Self-hosted NAS|Reliability & cost|
|Notifications|Stream + async workers|Sync DB writes on actions|Non-blocking UX|
|Likes counters|Cache increment + async sync|Every like = DB write|Reduce DB load|

---

## 14. Optional Enhancements

- Ranking feed with ML (engagement prediction).
    
- Stories (24h TTL) using TTL tables.
    
- Reels (short videos) with transcoding pipeline (FFmpeg workers).
    
- Hashtag/Explore using Search service (Elastic).
    
- Content moderation: AI image checks, report abuse flow.
    

---

## 15. Interview Wrap-up (How to Present)

1. Clarify features & scale.
    
2. Identify core components: Users/Follow, Posts/Media, Feed, Notifications.
    
3. Dive into **feed generation** (key complexity).
    
4. Discuss storage, caching, scalability & consistency.
    
5. Mention observability, security, trade-offs.
    
6. Close with potential improvements.
    

---

## 16. ASCII Cheat Sheet (Revision Page)

```
Users -> Post -> Fanout -> FeedStore -> Feed API -> Users
Media: Client -> S3 -> CDN -> Client
Follow edges: SQL/Cassandra
Cache: Redis for Feed & hot posts
Queue: Kafka for fanout & notifications
Hybrid push/pull feed to handle celebs
```

---
# Design Whatsapp
## 0. Problem Statement

Build a mobile-first chat app where users send/receive real-time messages (1:1 & groups), see delivery/read receipts, and work on flaky mobile networks.

---

## 1. Requirements

### Functional

- Register/login with phone or email.
    
- One-to-one and group chats (≤256 members, say).
    
- Send text, images, voice notes, small videos.
    
- Online/offline status, “typing…” indicator.
    
- Delivery states: sent ✓, delivered ✓✓, read ✓✓ blue.
    
- Message sync across devices, basic search.
    
- Optional (out of scope here): end-to-end encryption details, calls, stickers, story/status.
    

### Non-Functional

- Low latency (P95 send→deliver < 200 ms within same region).
    
- High availability (99.9%+).
    
- Mobile network tolerant (offline queueing, retries).
    
- Scalability: 50M DAU, peak 2M concurrent connections.
    
- Durability: no message loss.
    
- Privacy & security (TLS, ideally E2EE—mention but skip crypto deep dive unless asked).
    

---

## 2. Rough Sizing Assumptions

- Messages/day: 5B (≈ 57k/sec avg; 5× bursts).
    
- Avg message payload: text ~200B, media links ~1KB metadata (media big but goes to object storage/CDN).
    
- Stored history: keep 30 days hot (fast store), then archive.
    
- Concurrent connections: 2M WebSocket sessions (scale horizontally).
    

---

## 3. High-Level Architecture (ASCII)

```
        Mobile/Web Clients
                |
        +------------------+
        |  Edge Load Bal.  |
        +------------------+
                |
        +------------------+
        |  Gateway/API     |  <-- Auth, Rate limit
        +------------------+
          |        \
          |         \-----------------------+
   (REST gRPC)                              \
          v                                   v
+-------------------+                 +-------------------+
|  Messaging Svc    | <---- Pub/Sub -->|  Presence Svc     |
|  (Router/Fanout)  |                  |  (online/typing)  |
+-------------------+                  +-------------------+
    |       |   \                              |
    |       |    \                             v
    |       |     \                     +---------------+
    |       |      +--> Notification -->| Push Service  |
    |       |                            +---------------+
    |       v
    |   +------------------+      +---------------------+
    |   | Message Store    |      | Media Service       |
    |   | (Cassandra/Scylla|      | (Upload to S3/CDN)  |
    |   +------------------+      +---------------------+
    |             |
    +--> Search/Index (Elastic)  (async pipeline)
    |
    +--> Analytics/Spam (Kafka -> Spark/Flink -> DWH)
```

Long-lived connections (WebSocket/GRPC streams) exit at **Messaging Service**, which routes messages.

---

## 4. Data Model (Simplified)

```
User(id, phone, name, avatar_url, last_seen_ts)

Conversation(id, type: "direct"|"group", member_ids[], created_at)

Message(id (UUID), conv_id, sender_id, type:text/image/voice,
        body, media_url, ts_server, status_map{user_id:status}, ttl?)

DeliveryReceipt(msg_id, user_id, status: SENT|DELIVERED|READ, ts)

Presence(user_id, status: online/offline, last_seen_ts, typing_for_conv_id?)
```

(Use wide-column store: `PartitionKey = conv_id`, clustering by `ts` → efficient time-ordered reads.)

---

## 5. APIs (Examples)

```http
POST   /register              {phone, otp}
POST   /conversations         {members[]}
GET    /conversations?limit=...
GET    /messages?convId=..&cursor=..
POST   /messages               {convId, body, type}
POST   /receipts               {msgId, status}
GET    /presence/{userId}
WS     /connect (bi-dir stream for messages/events)
POST   /media/upload-url
```

---

## 6. Critical Flow: Send & Deliver Message

### Sequence (ASCII)

```
Sender Client
   |  send msg over WS
   v
Messaging Svc (entry shard)
   | assign msg_id, ts, persist to Message Store (write quorum)
   | publish to convo topic (Kafka/PubSub)
   v
Router/Fanout workers
   | fetch conv members
   | push msg to each online member's WS session
   | store "undelivered" flag for offline members
   v
Recipient Client(s)
   | receive msg, ACK receipt (DELIVERED)
   v
Messaging Svc updates receipt in DB, propagate to sender
   | when user reads: send READ receipt
```

---

## 7. Presence & Typing Indicators

- **Presence service** maintains in-memory state (Redis/Memory + gossip).
    
- Clients ping every N seconds; if miss deadlines → offline.
    
- Typing: short-lived events (`typing_start`, `typing_stop`) published to convo members via WS.
    

```
Client -> Presence Svc (ping)
Presence Svc -> publish presence events to subscribers (WS)
```

---

## 8. Storage Choices

- **Messages:** Cassandra/Scylla/Bigtable (append-only, high write throughput, time-sorted). Partition by conversation.
    
- **Receipts:** Same partition or separate table keyed by msg_id.
    
- **Media:** S3/GCS + CDN. Metadata in DB; clients upload via pre-signed URLs.
    
- **Search:** Elastic/OpenSearch indexing async (consume from Kafka).
    
- **Redis:**
    
    - Session tokens / WS connection map
        
    - Unread counts
        
    - Presence/typing ephemeral data
        
- **Cold Archive:** After 30/90 days, move to cheaper storage (S3 Glacier + index pointer).
    

---

## 9. Scaling & Sharding

- **Messaging Svc shards** by conversation_id hash → ensures users in same convo hit same shard (or use consistent hashing ring + stateful routers).
    
- **Cassandra** auto-shards by partition key (conv_id). Keep partition size reasonable (limit messages per partition; rotate conv partition IDs by time).
    
- **WebSocket scale:** Use load balancer + sticky sessions to a shard; or use a message broker (NATS/Kafka) for fanout to connection managers.
    

---

## 10. Consistency & Delivery Semantics

- **At-least-once delivery** to clients (idempotent message IDs prevent dupes).
    
- **Write path** uses quorum writes (Cassandra RF=3, W=2) to ensure durability.
    
- **Receipts are eventual;** message delivery status can lag slightly.
    
- **Ordering:** preserve per-conversation ordering by server timestamp; if multiple servers, sequence numbers per conv.
    

---

## 11. Offline & Retry Strategy

- Client queues outgoing msgs locally if offline.
    
- On reconnect, it resends unsent msgs (idempotent on msg_id).
    
- Server stores undelivered msgs → pushes when user reconnects (pull via sync API).
    
- Use “since_ts” sync endpoints to fetch missed events.
    

---

## 12. Security (Mention E2EE Briefly)

- **Transport security:** TLS for all connections.
    
- **Auth:** OAuth/JWT after OTP verification.
    
- **E2EE (advanced):** Double ratchet (Signal protocol). Server stores only cipher texts. Keys on client. (State this is extra and can be discussed if asked.)
    
- **Abuse checks:** limit media size, spam detection, report feature.
    

---

## 13. Observability & Ops

- Metrics: WS connections count, send→deliver latency, Cassandra write/read latency, queue lag.
    
- Tracing: per message path (OpenTelemetry).
    
- Logs: sampled to control volume.
    
- Alerts: surge in undelivered msgs, partition hot-spot, WS error rate.
    

---

## 14. Trade-offs & Alternatives

|Topic|Choice|Alternative|Why|
|---|---|---|---|
|Protocol|WebSocket/GRPC streams|Long-polling|WS reduces overhead for real-time|
|Store|Cassandra wide-rows|SQL sharding / Mongo|High write throughput, linear scale|
|Fanout|Server-side per conv|Client pull every N seconds|Real-time UX vs polling overhead|
|Ordering|Server seq per conv|Client timestamp reconcile|Simpler consistency model|
|Receipts|Async updates|Sync on every read|Lower latency in hot path|
|Presence storage|Redis/in-memory + heartbeats|DB per update|Low latency, high churn data|

---

## 15. Extensions / Future

- Voice/video calls (WebRTC signaling service).
    
- Stickers/GIF search service.
    
- Message reactions, thread replies.
    
- Multi-device sync (complicated key sync in E2EE).
    
- Large group (>1k) broadcast channels (optimize fanout differently).
    
- Message recall/delete for everyone.
    

---

## 16. Interview Wrap-up Cheat Sheet

```
1) Clarify: 1:1 + group chat, receipts, presence
2) Scale: 5B msgs/day, 2M concurrent
3) Architecture: WS gateway -> Messaging svc -> Cassandra + Pub/Sub
4) Flows: send msg, deliver, receipts, offline sync
5) Presence/typing ephemeral via Redis
6) Consistency: per-conv ordering, at-least-once delivery
7) Trade-offs & failure handling
8) Mention E2EE & future features
```

---
# Design Google Drive
## 0. Problem Statement

Build a service where users upload, store, and sync files across devices, share them with others, and view version history.

---

## 1. Requirements

### Functional

- User auth, create folders/files.
    
- Upload/download files (web & desktop/mobile clients).
    
- Automatic client sync (detect local changes, push to cloud, pull remote changes).
    
- File versioning & conflict resolution.
    
- Sharing: link-based & permission-based (read/write).
    
- Search by filename and (optional) content.
    

### Non‑Functional

- High durability (11+ 9’s), availability (99.9%+).
    
- Efficient bandwidth usage (chunking, delta sync).
    
- Low upload/download latency, but sync can be eventually consistent.
    
- Scale to hundreds of millions of files and users.
    
- Security: encryption at rest & in transit.
    

_Assumptions (for sizing talk):_

- 10M MAU, avg 5 GB/user → 50 PB total data.
    
- Avg file 2 MB, but long tail (GB videos).
    
- 200K concurrent active sync clients.
    
- 1M file change events/day (~11/s avg, spikes 10×).
    

---

## 2. High-Level Architecture (ASCII)

```
Clients (Desktop Agents / Mobile / Web)
        |            ^
        v            |
+---------------------------+
|  Sync & API Gateway      |
|  (REST/gRPC/WebSocket)   |
+------------+--------------+
             |
             v
      +--------------+            +----------------------+
      | Metadata Svc |<---------->|  Auth / ACL Service  |
      +--------------+            +----------------------+
             |
             v
   +--------------------+         +----------------------+
   | Metadata Store     |         | Notification/Queue   |
   | (SQL + Cache)      |         | (Kafka/SNS/SQS)      |
   +--------------------+         +----------------------+
             |                                |
             | change events                  | fanout
             v                                v
    +-------------------+            +-------------------+
    | Chunking/Dedup Svc|            | Sync Orchestrator |
    +-------------------+            +-------------------+
             |                                |
             v                                v
   +----------------------+          +-----------------------+
   | Object Storage (S3)  | <------> | CDN (downloads)       |
   +----------------------+          +-----------------------+
             ^
             |
      +-------------------+
      | Virus Scan / DLP  |
      +-------------------+

Search Pipeline:
Queue -> Indexer -> Search Service (Elastic/OpenSearch)
```

---

## 3. Data Model (Simplified)

```
User(id, email, hashed_pw, quota, created_at)

FileEntry(id, user_id, parent_folder_id, name, is_folder, latest_version_id, created_at)

FileVersion(id, file_id, chunk_list[], size, md5, created_at)

Chunk(id, checksum_sha256, size, storage_path, ref_count)

ShareLink(id, file_id/folder_id, owner_id, token, perms, expiry)

ACL(resource_id, subject_id (user/group), permission)
```

Indexes:

- FileEntry: (user_id, parent_folder_id, name) unique.
    
- Chunk: checksum for dedup.
    
- FileVersion: file_id index.
    
- ACL/ShareLink: token unique index.
    

---

## 4. Core APIs

```http
POST   /files/upload-init {parentId, name, size}
PUT    /files/upload-chunk?uploadId=..&chunkNo=..   (stream data)
POST   /files/upload-complete {uploadId}
GET    /files/{fileId}/download-url

GET    /tree?parentId=...
POST   /folders
DELETE /files/{id}
GET    /files/{id}/versions
POST   /share/link {fileId, perms, expiry}
GET    /search?q=...

WS     /sync/subscribe  (server push for file change events)
```

Client sync protocol (desktop agent):

- Poll/WS for changes since last cursor.
    
- Local file watcher triggers upload for modified files.
    

---

## 5. Upload / Sync Flow

### Upload (Chunked & Dedup)

```
Client -> upload-init
  | server returns chunk size, uploadId
Client -> for each chunk:
          - compute hash
          - ask server "haveChunk(hash)?"
             yes -> skip upload
             no  -> upload chunk to S3 pre-signed URL
Client -> upload-complete (list of chunk hashes)
Server -> create FileVersion (chunks list) & update metadata
Server -> push "file_changed" event to queue
```

ASCII sequence:

```
Client ---init---> API
Client --chunk hash--> ChunkSvc -> found? skip : uploadURL
Client ---PUT chunk--> S3
Client ---complete--> API -> Metadata DB
API -> Queue "file_changed"
Sync Orchestrator -> notify other devices via WS
```

### Sync (Pull Remote Changes)

```
1. Client connects (WS) & sends lastCursor.
2. Server streams events since cursor (create/update/delete).
3. Client downloads new/updated files via CDN.
4. Update local cursor.
```

---

## 6. Chunking & Dedup Strategy

- Fixed size (e.g., 4 MB) or variable (Rabin fingerprinting) to improve delta detection.
    
- Hash each chunk (SHA-256). Store in Chunk table; if exists, increase ref_count.
    
- Store chunks in object storage using content-addressable paths.
    
- Benefits: Saves bandwidth & storage for repeated content/versioning.
    

---

## 7. Conflict Resolution

Cases: Two devices edit same file offline.  
Approach:

- On upload, compare base version vs latest server version.
    
- If mismatch → conflict: store both versions; mark one as “conflict copy (device name timestamp)”.
    
- Optionally merge for text files (not for binaries).
    

---

## 8. Storage & Indexing

- **Metadata:** Relational DB (Postgres/Aurora) for transactions (file tree ops). Scale via sharding by user_id + read replicas. Cache hot paths in Redis.
    
- **Chunks & Versions:** Object storage (S3/GCS). Lifecycle policies to move old versions to cheaper tiers (Glacier).
    
- **Search:** Index file/folder names + maybe content (OCR/PDF parse) into Elastic.
    
- **Queues:** Kafka/SNS for event distribution (sync notifications, search indexer, audit logs).
    
- **Virus Scanning / DLP:** Lambda/worker triggered on new chunks.
    

---

## 9. Caching & CDN

- Serve downloads via CDN edge (signed URLs).
    
- Metadata read caching in Redis to reduce DB load.
    
- Client-side local cache for unchanged files.
    

---

## 10. Scalability & Partitioning

- **API & Sync Gateway:** Stateless, scale horizontally (k8s + autoscaling).
    
- **Metadata DB:** Shard by user_id; use consistent hashing. Foreign keys constrained within shard.
    
- **Chunk store:** Object storage auto-scales. Maintain a small service to manage ref_counts and chunk GC.
    
- **Notification fanout:** Partition topics by user_id.
    
- **Search indexing:** Distributed consumers of queue.
    

---

## 11. Consistency & Reliability

- **Metadata ops:** Use transactions to keep file tree consistent.
    
- **Upload finishing:** Two-phase (init → chunks → commit). If commit fails, mark orphaned chunks for GC.
    
- **Eventual consistency:** Other devices may see changes after small delay. Acceptable for sync.
    
- **Idempotency:** upload-complete with same uploadId should not duplicate versions.
    

---

## 12. Security

- TLS for all traffic.
    
- Encryption at rest: S3 SSE/KMS; metadata DB encryption.
    
- Access control: verify ACL/share link token on every download.
    
- Signed URLs (short TTL) for direct S3/CDN access.
    
- Audit logs for admin actions & sharing; GDPR delete.
    

---

## 13. Observability

- Metrics: upload success rate, average chunk dedup %, metadata DB latency, queue lag, sync delay.
    
- Tracing: request path (OpenTelemetry).
    
- Logs: structured; sample large uploads.
    
- Alerts: spike in 5xx, drop in dedup ratio (bug), backlog in notifications.
    

---

## 14. Trade-offs & Alternatives

|Topic|Choice|Alternative|Why|
|---|---|---|---|
|Chunk size|Fixed 4 MB|Variable chunking|Simpler, good enough dedup|
|Metadata store|SQL sharded by user|NoSQL Doc store|Strong consistency & transactions|
|Sync notif|WS + cursor|Long polling|Lower latency, less overhead|
|Dedup store|Global chunk store|Per-user dedup|Better space saving, but heavier GC|
|Conflict policy|Keep both versions|Last-write-wins|Avoid silent data loss|

---

## 15. Extensions / Future Work

- Real-time collaborative editing (Google Docs style—CRDT/OT).
    
- Delta-sync within files (binary diffs).
    
- Offline sharing via LAN sync between devices.
    
- Advanced search (content OCR, image labels).
    
- Ransomware detection (sudden mass encrypt & upload).
    
- Quota management, billing service.
    

---

## 16. Interview Wrap-Up Cheat Sheet

```
1) Clarify features: upload, sync, share, versions
2) Traffic & size assumptions
3) Core pieces: Metadata DB, Chunk store, Sync/Notif pipeline
4) Flows: chunked upload + dedup, sync subscribe/pull
5) Consistency & conflicts
6) Security & observability
7) Trade-offs
8) Future enhancements
```

---
# Design Youtube
## 0. Problem Statement

Build a platform where creators upload videos. Users browse/search, stream videos (adaptive bitrate), like/comment/subscribe, and get recommendations.

---

## 1. Requirements

### Functional

- Upload videos (large files), process/transcode to multiple qualities.
    
- Store & stream videos via CDN (HLS/DASH).
    
- Browse/search videos, channels, categories.
    
- Watch page: player, likes, comments, views counter.
    
- Subscriptions & notifications.
    
- Recommendation feed (“Home”, “Up next”).
    
- Optional/out of scope: live streaming, ads, DRM.
    

### Non‑Functional

- High write cost for upload/transcode; extremely high read (stream) traffic.
    
- Low startup delay (<2s) & smooth playback (ABR).
    
- 99.9% availability, global distribution.
    
- Durability for original uploads (11x9s).
    
- Eventually consistent counters/metrics OK.
    

---

## 2. Rough Sizing (assume)

- 20M DAU, 3M daily video plays, peak 100k concurrent streams.
    
- Uploads: 50k videos/day, avg 500MB (orig).
    
- Store originals + ~6 encoded renditions (144p–1080p).
    
- Peak stream bitrate ~3 Mbps (HD), average 1 Mbps.
    
- Metadata small (KBs), but logs huge (TB/day).
    

---

## 3. High-Level Architecture

```
Creators (Upload)          Viewers (Stream)
      |                           |
      v                           v
+-----------------+        +------------------+
| Upload Service  |        | CDN (Edge POPs)  |
+-----------------+        +------------------+
      |                           ^
      v                           |
+----------------------+     +---------------------+
| Object Storage (orig)|     | Streaming Gateway   |
+----------------------+     +---------------------+
      |                           ^
      v                           |
+----------------------+     +----------------------+
| Transcode Pipeline   |-->  | Manifest/Chunk Store |
| (FFmpeg workers)     |     | (S3/GCS)             |
+----------------------+     +----------------------+
      |
      v
+----------------------+
| Metadata Service     |
+----------------------+
      |
      v
+----------------------+     +---------------------+
| Metadata DB (SQL)    |<--->| Search Service      |
+----------------------+     | (Elastic/OpenSearch)|
      |                       +---------------------+
      v
+----------------------+     +---------------------+
| Reco/Analytics Svc   |<--->| Event Stream (Kafka)|
+----------------------+     +---------------------+
```

---

## 4. Core Data Model (simplified)

```
User(id, name, email, ...)

Channel(id, user_id, title, desc, subs_count, ...)

Video(id, channel_id, title, desc, tags[], duration, status,
      upload_ts, views, likes, manifest_path, thumbnails[])

TranscodeJob(id, video_id, status, profiles[], created_at, updated_at)

Comment(id, video_id, user_id, text, ts)

Subscription(user_id, channel_id, created_at)
```

Indexes:

- Video: (title, tags) full-text; (channel_id, upload_ts) for listing.
    
- Comment: video_id index.
    
- Subscription: user_id index.
    

---

## 5. Key APIs

```http
POST /videos/upload-init     -> {uploadUrl}
POST /videos/{id}/metadata
GET  /videos/{id}            -> metadata + manifest URLs
GET  /videos?search=...&tag=...

POST /channels/{id}/subscribe
GET  /users/{id}/feed

POST /videos/{id}/like
POST /videos/{id}/comment
GET  /videos/{id}/comments?cursor=...
```

Internal:

- `/transcode/start`, `/transcode/status`
    
- Event topics: `video_uploaded`, `video_transcoded`, `video_played`, `like_added`
    

---

## 6. Critical Flows

### A) Upload → Transcode → Publish

```
Client -> Upload Svc: request upload URL
Client -> PUT original file to Object Storage (S3)
Upload Svc -> create Video record (status=UPLOADING)
Upload Svc -> enqueue TranscodeJob (Kafka)
Transcoder Workers -> pull job
    - download original
    - transcode into renditions (240p,360p,480p,720p,1080p) -> chunk + manifest (HLS/DASH)
    - store chunks/manifests in storage
    - capture thumbnails
    - update Video(status=READY, manifest_path)
    - send event "video_transcoded"
```

ASCII:

```
Uploader --> UploadSvc --> S3(orig)
                      \--> Kafka(Job)
Kafka -> Transcoder -> S3(encoded) -> Video DB READY
```

### B) Playback (Adaptive Streaming)

```
Player requests /video/{id}
  API returns metadata + manifest URL
Player fetches manifest (.m3u8/.mpd) from CDN
CDN serves segment chunks (.ts/.m4s) from edge cache
Player switches bitrate based on bandwidth
Player sends play/pause/finish events -> Event Stream
```

ASCII:

```
Player -> API -> manifest URL
Player -> CDN (manifest + chunks)
CDN miss -> origin storage -> edge cache
```

---

## 7. Streaming Details

- Use **HLS/DASH**: manifest lists multiple bitrates; segments ~2–6s.
    
- **CDN** caches segments globally; origin is object storage.
    
- For startup, push first segments closer (pre-warm) or use shorter first chunk.
    
- DRM (optional): Widevine/FairPlay; license server.
    

---

## 8. Storage & Caching

- **Object Storage (S3/GCS):** Originals, encoded chunks, manifests, thumbnails.
    
- Lifecycle rules to move cold originals to Glacier.
    
- **CDN:** Akamai/CloudFront/Cloudflare for edge caching.
    
- **Metadata DB:** SQL (Postgres/Aurora) with read replicas; cache hot video meta in Redis.
    
- **Search Index:** Elastic; ingest from DB via pipeline.
    
- **Event Logging:** Kafka -> DWH (BigQuery/Redshift) for analytics/reco.
    

---

## 9. Recommendations (simple view)

- Collect watch history and implicit signals (view duration, likes).
    
- Batch jobs (Spark) compute candidate videos (collaborative filtering).
    
- Online service re-ranks using freshness/popularity.
    
- Cache per-user home feed.
    

---

## 10. Scalability & HA

- Stateless services in k8s (auto-scale).
    
- Transcoder pool scales with upload backlog (spot instances).
    
- DB: shard by video_id/channel_id if needed.
    
- Kafka cluster for events (multi-broker).
    
- Multi-region: replicate manifests/chunks; geo-DNS to nearest CDN POP.
    

---

## 11. Consistency & Counters

- `views`, `likes` counters updated asynchronously:
    
    - Client fires event → Kafka → counter service increments in Redis + periodic batch flush to DB.
        
- Comments strongly consistent per video (SQL transaction).
    
- Video status transitions via state machine.
    

---

## 12. Security

- Auth tokens (JWT).
    
- Upload URLs are pre-signed short TTL.
    
- DRM/tokens on manifests if premium content.
    
- Abuse detection: scan uploads (hash match, AI).
    
- Rate limiting on comments/likes.
    

---

## 13. Observability

- Metrics: startup latency, rebuffer ratio, CDN hit %, transcode job latency, DB QPS.
    
- Tracing: upload & playback request path.
    
- Logs: sampled for playback, full for errors.
    
- Alerts: spike in 5xx, drop in CDN hit %, transcode backlog.
    

---

## 14. Trade-offs (cheat sheet)

|Area|Choice|Alternative|Reason|
|---|---|---|---|
|Streaming|HLS/DASH + CDN|Progressive download|ABR, better QoE|
|Transcoding|Async pipeline (FFmpeg farm)|Sync on upload|Non-blocking, scalable|
|Counters|Async increment via Kafka|Sync DB update|Reduce DB hot spots|
|Metadata DB|SQL + cache|NoSQL only|Transactions & joins|
|Search|Elastic|DB LIKE queries|Relevance, scale|

---

## 15. Future Enhancements

- Live streaming (RTMP ingest → HLS).
    
- Ads insertion (client/server-side).
    
- Personalized home feed with deep ML.
    
- Offline download (encrypted segments).
    
- Comment moderation (NLP).
    
- Multi-CDN switching for resilience.
    

---

## 16. Interview Wrap-up (Quick Recap)

```
1. Clarify: upload, transcode, stream, search, reco
2. Size assumptions (uploads, playbacks)
3. Architecture: upload->transcode->store->CDN; metadata & search
4. Flows: upload/transcode, playback, counters
5. Scalability: CDN, async pipeline, shards
6. Consistency: async counters, strong for meta
7. Trade-offs + future work
```

---

# Design Uber

## 0. Problem Statement

Build a platform where riders request trips, nearby drivers get matched, both track each other in real time, fare is calculated/paid, and trip history is stored.

---

## 1. Requirements

### Functional

- Rider: signup/login, request/cancel ride, real‑time driver ETA, pay, rate.
    
- Driver: go online/offline, accept/reject, navigation, earnings, rate rider.
    
- Matching: pick best driver (distance, ETA, rating, car type).
    
- Pricing: base + distance + time (+ surge).
    
- Trip lifecycle: request → dispatch → accept → pickup → dropoff → payment.
    
- Notifications (push/SMS).
    
- Admin/ops dashboards (fraud, support).
    

### Non‑Functional

- Low latency: match within ~2–3 s; location updates every 2–5 s.
    
- High availability (99.9%); global scale.
    
- Accurate, scalable location indexing.
    
- Strong consistency for money flows; eventual OK for analytics.
    
- Security (PII, payments).
    

---

## 2. Rough Sizing (assume)

- 10M DAU riders, 1M DAU drivers.
    
- 500k concurrent users, 50k live trips.
    
- Location updates: each active user every 3 s → ~170k updates/sec peak.
    
- Trip requests ~5k/sec peak.
    
- Store billions of trip records (metadata ~2–3 KB).
    

---

## 3. High-Level Architecture

```
Clients (Rider/Driver apps)
        |
        v
+--------------------+
| API Gateway        |
+--------------------+
   |   |      \
   |   |       \---- Auth/Rate Limit
   |   |
   |   +--> User/Driver Service  --> SQL DB (users, cars, docs)
   |
   +--> Trip Service  -----------> Trip DB (SQL/Cassandra)
   |
   +--> Matching Service --------> Geo Index (Redis/ElasticGeo)
   |                \
   |                 \--> Pricing Service (surge, fare calc)
   |
   +--> Location Service -------> In-memory/Redis + Kafka (stream)
   |
   +--> Payment Service --------> Payment Gateway, Ledger DB
   |
   +--> Notification Service ---> Push/SMS/Email
   |
   +--> Analytics Stream (Kafka) -> DWH/ML
```

**Realtime comms:** WebSockets/GRPC streams for location & status pushes.

---

## 4. Core Data Model (simplified)

```
User(id, name, phone, type: rider/driver, rating, ...)
DriverStatus(driver_id, online/offline, car_type, last_loc)

Trip(id, rider_id, driver_id, status,
     start_loc(lat,lng), end_loc(lat,lng),
     start_ts, end_ts, distance_km, duration_sec,
     fare, surge_multiplier, payment_status)

GeoCell(cell_id, driver_ids[])   // ephemeral index of available drivers

PaymentTxn(id, trip_id, amount, currency, method, status, created_at)

Rating(id, trip_id, rater_id, ratee_id, stars, comment)
```

Indexes:

- Trip by rider_id/driver_id + time.
    
- Geo index by cell_id.
    
- PaymentTxn by trip_id.
    

---

## 5. Key APIs

```http
POST /rider/ride/request   {pickup, dropoff, type}
POST /rider/ride/cancel
GET  /rider/trips?cursor=...

POST /driver/status        {online:true/false}
POST /driver/location      {lat,lng, ts}
POST /driver/ride/accept   {tripId}
POST /driver/ride/complete {tripId, odometer, time}

POST /payment/charge       {tripId, token}
POST /rating               {tripId, stars, comment}

WS  /realtime/updates  (driver & rider)
```

---

## 6. Critical Flows

### A) Rider Requests Ride → Match Driver

```
Rider App -> Trip Svc: request(pickup, dropoff)
Trip Svc -> Matching Svc: find driver
Matching Svc:
   - Query Geo Index for nearby drivers
   - Rank by ETA, rating, car type
   - Reserve first driver (optimistic lock)
   - Send offer to Driver app (push/WS)
Driver -> accept
Matching Svc -> confirm assignment
Trip Svc -> update trip.status=ASSIGNED
Notify Rider (driver info, ETA)
```

ASCII:

```
Rider -> TripSvc -> MatchSvc -> GeoIndex
                          |-> Driver (offer)
Driver -> MatchSvc (accept)
MatchSvc -> TripSvc -> Rider/Driver notify
```

### B) Realtime Location Updates

```
Driver App -> Location Svc (WS/HTTP) every 3 s
Location Svc -> update Redis/Memory + publish to Kafka
Matching Svc subscribes for up-to-date locations
Trip Svc pushes driver location to Rider via WS
```

ASCII:

```
Driver -> LocSvc -> Redis/Kafka
                   ^          \
                   |           -> TripSvc -> Rider WS
                   +-> MatchSvc (reads Redis)
```

### C) Trip Completion & Payment

```
Driver marks complete (or auto by geofence)
Trip Svc calculates final fare (time+distance+surge)
Payment Svc charges card/wallet
Payment status -> Trip Svc
Both rate each other
```

---

## 7. Geo Indexing & Matching

- Divide world into grid cells (e.g., Geohash level 6/7).
    
- Maintain **available drivers** per cell in Redis (sorted by timestamp or load).
    
- To find nearby: take rider cell + neighbors, compute ETA with road graph (optional: external routing API).
    
- Locking/reservation: set a short TTL key `driver:{id}:locked` to avoid double-assign.
    

---

## 8. Pricing & Surge

- Base fare + per km + per minute.
    
- Surge multiplier based on **demand/supply ratio** per region cell.
    
- Pre-calc surge every N seconds with streaming data (Kafka → Flink).
    
- Final fare recalculated after trip using actual metrics.
    

---

## 9. Storage & Streams

- **Redis/In-memory:** live locations, geo cells, driver availability.
    
- **Kafka/PubSub:** location events, trip events, pricing signals, analytics.
    
- **SQL (Postgres/Aurora):** users, trips (hot), payments, ratings. Partition trips by month/user_id.
    
- **Cassandra/Bigtable (optional):** very high write trip events/history.
    
- **Ledger DB (ACID):** payment records, settlement.
    

---

## 10. Consistency & Reliability

- Trip state transitions (FSM): REQUESTED → DRIVER_OFFERED → ASSIGNED → STARTED → COMPLETED → PAID. Enforce via single writer or transactions.
    
- Location updates: eventual consistency OK.
    
- Payments: **strong consistency / idempotency** (retry-safe tokens, transaction records).
    
- If driver rejects/no response: timeout & rematch.
    
- If Redis down: fallback to slower DB geo query (degraded mode).
    

---

## 11. Security

- OAuth/JWT for APIs; TLS everywhere.
    
- Validate location spoofing (anomaly detection).
    
- Mask phone numbers via relay service.
    
- PCI compliance for card data (tokenize via gateway).
    
- Audit logs for fare changes, manual adjustments.
    

---

## 12. Observability

- Metrics: match latency, acceptance rate, cancel rate, ETA accuracy, surge ratio, payment failures.
    
- Logs: trip state transitions, driver online events.
    
- Tracing: request across Trip→Match→Payment.
    
- Alerts: spike in unmatched requests, Redis miss %, Kafka lag.
    

---

## 13. Trade-offs

|Topic|Choice|Alternative|Why|
|---|---|---|---|
|Location store|Redis + Kafka|Only DB|Low-latency, high QPS updates|
|Geo search|Geohash grid + Redis sets|R-tree in SQL|Simpler, faster in-memory|
|Match strategy|Nearest ETA + lock|Broadcast to many drivers|Reduce spam/notification load|
|Payments|Async confirm after trip|Pre-authorize full amount|UX vs certainty; can pre-auth too|
|Surge calc|Stream processing|Batch hourly|Real-time demand spikes|

---

## 14. Extensions / Future Work

- Pooling (UberPool): route optimization with multiple pickups.
    
- Scheduled rides (cron-like matching).
    
- Multi-stop trips.
    
- Driver incentives & gamification service.
    
- Fraud detection ML pipeline (fake trips, GPS spoofing).
    
- In-app chat/call (VoIP).
    
- Multi-region replication & failover.
    

---

## 15. Interview Wrap-Up Cheat Sheet

```
1) Clarify: rider/driver, match, real-time, payment
2) Estimate traffic (trip reqs, loc updates)
3) Arch: TripSvc, MatchSvc, LocationSvc, PaymentSvc, Redis+Kafka
4) Flows: request->match, location updates, payment
5) Geo indexing (geohash), surge, locking
6) Consistency (trip FSM), idempotent payments
7) Trade-offs, failures, future features
```

---
# Design Google Docs
## 0. Problem Statement

Multiple users edit the same document simultaneously and see each other’s changes in near real time, with version history and offline sync.

---

## 1. Requirements

### Functional

- Create/open/share documents (permissions: owner/editor/viewer).
    
- Real-time collaborative editing (multiple cursors, selections).
    
- Conflict-free merges (no lost updates).
    
- Presence indicators (who’s online, cursor position).
    
- Commenting & basic formatting (bold/italic/headers).
    
- Version history & “undo/redo”.
    
- Offline editing → sync on reconnect.
    

### Non‑Functional

- Low latency: local echo instantly, remote updates <150 ms.
    
- High availability; data durability.
    
- Support thousands of concurrent docs, ~50 editors/doc (typical).
    
- Strong security (doc ACL, encryption in transit).
    
- Consistency model: eventual but convergent (all see same doc state).
    

_Assumptions_: 5M DAU, 200k concurrent editors, avg doc size ~100 KB (plain text), edits ~5 ops/sec/doc peak.

---

## 2. Core Challenge: Concurrency Control

Two mainstream approaches:

1. **Operational Transformation (OT)**: Transform incoming ops against concurrent ones to preserve intent.
    
2. **CRDTs (Conflict-free Replicated Data Types)**: Data structure designed to converge automatically (e.g., RGA, Yjs).
    

**Pick one** (mention both). For interview, choose **OT** (classic in Google Docs) or **CRDT** (modern, simpler to reason distributedly).

---

## 3. High-Level Architecture

```
Clients (Web/Mobile/Desktop)
        |  WebSocket (bi-dir)
        v
+------------------------+
|  Realtime Collaboration|
|  Gateway (WS servers)  |
+------------------------+
        |
        v
+------------------------+       +--------------------+
|  Doc Session Service   |<----->| Presence Service   |
|  (OT/CRDT engine)      |       | (cursors, users)   |
+------------------------+       +--------------------+
        |
        v
+------------------------+       +-------------------+
|  Persistence Service   |<----->| Version/History   |
|  (Doc store + cache)   |       | Service           |
+------------------------+       +-------------------+
        |
        v
+------------------------+
|  Storage (DB/Object)   |
+------------------------+

Async:
Event Stream (Kafka) -> Search Indexer, Analytics, Backup
```

---

## 4. Data Model (Simplified)

```
Document(id, owner_id, title, created_at, acl[])

DocState(id=doc_id, current_version, snapshot_path)

Operation(id, doc_id, user_id, base_version, op_payload, ts)

Comment(id, doc_id, user_id, range(start,end), text, ts)

Presence(doc_id, user_id, cursor_pos, selection_range, last_ping_ts)
```

- **OT op_payload** example: “insert ‘abc’ at position 42”, “delete 3 chars at 15”.
    
- **CRDT** state: sequence of character IDs (id = (counter, replica_id)).
    

Indexes:

- Document by owner/shared.
    
- Operation by doc_id & version.
    
- Presence ephemeral (in-memory store).
    

---

## 5. Key APIs / Protocols

**HTTP (REST) for CRUD & sharing:**

```http
POST /docs          {title}
GET  /docs/{id}
POST /docs/{id}/share {userId, role}
GET  /docs/{id}/history?cursor=...
```

**WebSocket for realtime:**

- `join_doc` {docId, lastKnownVersion}
    
- `op` {docId, clientSeq, baseVersion, payload}
    
- `presence` {cursorPos, selection}
    
- `ack` {clientSeq, committedVersion}
    
- `remote_op` (server → clients)
    

---

## 6. Critical Flows

### A) Client Edit → Broadcast

```
Client types "Hi"
  |
  v
1. Apply locally (optimistic)
2. Send op over WS: baseVersion=42, insert("Hi", pos=10)
3. Server receives op, transforms against unseen ops (OT) or merges (CRDT)
4. Server increments version → 43, persists op
5. Server broadcasts transformed op to other clients
6. Clients integrate op into their state; update cursors
```

ASCII sequence:

```
Client -> (op v42) -> Collab Server
Collab Server: transform/merge -> v43
Collab Server -> persist op -> broadcast to others
Others -> apply op
Sender -> ack (confirm v43)
```

### B) New User Joins Existing Doc

```
Client -> join_doc(docId, lastVersion=0)
Server -> send snapshot (or patch from lastVersion)
Client -> apply, now at currentVersion
Server -> start streaming new ops
```

### C) Offline → Reconnect

```
Client offline: buffer ops with local seq
Reconnect: send ops batch with baseVersion
Server transforms/merges; resolves conflicts
Client gets ack & any missed remote ops
```

---

## 7. OT vs CRDT Quick Notes

|Aspect|OT|CRDT|
|---|---|---|
|Convergence|Needs transform functions|Guaranteed by design|
|Complexity|Transform functions per op type|More memory overhead (IDs)|
|Latency|Central server simplifies OT|P2P possible, but server common|
|Industry use|Google Docs, Etherpad|Figma, Notion (Yjs/Automerge)|

In interview: explain one deeply, mention the other.

---

## 8. Persistence & Snapshots

- Store every op → replay to reconstruct doc (costly).
    
- Periodic **snapshots** (e.g., every N ops or 5 minutes): persist full doc text/structure to object storage.
    
- On load: snapshot + ops after snapshot = current state.
    
- Version history: pointer to snapshot + op range.
    

---

## 9. Presence, Cursors & Comments

- Presence service keeps ephemeral states in Redis or in-memory map (TTL).
    
- Broadcast cursor/selection updates throttled (e.g., 2–4/s).
    
- Comments stored in DB; anchored to character ranges (need to update ranges as text changes -> track with op transforms).
    

---

## 10. Scaling & Sharding

- **Session sharding**: route all WS for a doc to same “doc session” node (sticky by doc_id hash).
    
- **Backpressure**: If >N active editors, split doc into sections or raise limits.
    
- **Storage**:
    
    - Ops in append-only store (Cassandra/Kafka log + DB).
        
    - Snapshots in object store (S3).
        
- **Cache**: Redis for hot snapshots.
    

---

## 11. Consistency & Fault Tolerance

- **Server as authority** for version numbers (linearizable per document).
    
- If doc session node crashes:
    
    - Rebuild state from last snapshot + ops (warm standby or stateless compute).
        
    - Clients reconnect; some ops may need retransmission (idempotent via clientSeq).
        
- Exactly-once apply via operation IDs.
    

---

## 12. Security

- ACL checks on every join/op.
    
- TLS WS connections; JWT auth.
    
- Data encryption at rest.
    
- Audit logs for doc access/changes.
    
- Rate limiting to avoid flooding (DoS).
    

---

## 13. Observability

- Metrics: ops/sec, transform latency, broadcast latency, reconnect rate.
    
- Tracing: WS message path (OpenTelemetry).
    
- Logs: sample ops (size, type), errors.
    
- Alerts: spike in missed acks, doc rebuilds, transform timeouts.
    

---

## 14. Trade-offs Cheat Sheet

|Topic|Choice|Alternative|Why|
|---|---|---|---|
|Concurrency|OT w/ central server|CRDT|Simpler to reason, widely known|
|Transport|WebSocket bi-dir|Long-polling|Lower latency, push-friendly|
|Storage|Ops + periodic snapshots|Full overwrite per save|Efficient, keeps history|
|Presence|Redis/in-memory ephemeral|SQL persistence|Low latency, high churn|
|Offline sync|Buffer ops locally, transform|Lock doc (no offline edits)|Better UX|
|Versioning|Incremental version numbers|Vector clocks|Simpler for centralized system|

---

## 15. Future Enhancements

- Rich text & embedded objects (images, tables) → tree CRDT/OT.
    
- Real-time cursors colors, avatars, chat side panel.
    
- Permissions per section of doc.
    
- Comment threads & suggestions mode (“track changes”).
    
- Export to PDF/Word.
    
- ML-powered grammar/style suggestions.
    

---

## 16. Interview Wrap-Up (Cheat Sheet)

```
1. Clarify: real-time edits, multi-user, versioning, offline
2. Consistency approach: OT or CRDT (pick one, mention other)
3. Arch: WS gateway -> Doc session svc -> storage (ops + snapshots)
4. Flows: edit op, join doc, offline sync
5. Presence & comments (ephemeral vs persistent)
6. Scaling: shard by doc, snapshots, caching
7. Fault tolerance, security, observability
8. Trade-offs & future features
```

---
# Revision Sheet
## A. 10‑Step Blueprint (use for ANY system design)

1. **Clarify scope & constraints** (features, users, regions).
    
2. **Traffic & sizing** (R/W QPS, storage, latency SLOs).
    
3. **APIs & data model (high-level)**.
    
4. **Core flows / sequence** (happy path first).
    
5. **High-level architecture diagram** (clients → gateway → services → DB/queue/cache).
    
6. **Storage choices & partitioning**.
    
7. **Caching, CDN, queues, search** (where & why).
    
8. **Scaling & HA** (sharding, replicas, autoscale, multi-region).
    
9. **Consistency, fault tolerance, idempotency**.
    
10. **Security, observability, trade-offs, future work**.
    

Keep each section to 1–3 lines when answering fast.

---

## B. Fast Sizing Cheats

- **Throughput:** QPS = (events/day) ÷ 86,400. Peak ≈ 5–10× avg.
    
- **Storage/Year:** events/day × size × 365. Keep “hot” vs “cold”.
    
- **Cache hit % target:** 80–95%.
    
- **Latency targets (P95):** API read <150ms, write <300ms, background (async) secs/mins.
    
- **Judge/ML jobs:** talk “queue depth” & “worker autoscale”.
    
- **WS connections per node:** typically 20k–50k (depends on infra).
    
- **Shard key rule:** Cardinality high, uniform distribution, low cross-shard ops.
    

---

## C. Building Blocks Quick List

- **Gateway:** auth, rate limit, routing.
    
- **Services:** user, content, search, feed, messaging, payment. Stateless.
    
- **DB:** SQL (transactions/relational), NoSQL KV (scale, wide rows), Columnar (analytics), Graph (relationships).
    
- **Cache:** Redis/Memcached; patterns: read-through, write-through, write-behind, TTL + version keys.
    
- **Queue/Stream:** Kafka/SQS/PubSub for decoupling, retries, backpressure, event sourcing.
    
- **Search:** Elastic/OpenSearch for full-text & ranking.
    
- **Blob store:** S3/GCS/MinIO for big binaries; pre-signed URLs.
    
- **Compute isolation:** containers vs microVMs (Firecracker).
    
- **Realtime:** WebSocket/gRPC streaming, SSE.
    
- **CDN:** static media, HLS video segments, problem statements.
    

---

## D. Consistency Patterns

- **Strong:** Payments, order booking, submission state.
    
- **Eventual:** Counts, feeds, notifications, analytics.
    
- **Idempotency keys** for POSTS.
    
- **Exactly-once feel:** “At-least-once + idempotent consumer”.
    
- **Leader election & leases** for workers.
    

---

## E. Go‑To Trade-offs (memorize a few)

|Topic|Option A|Option B|Pick Reason|
|---|---|---|---|
|Feed|Push (fanout-on-write)|Pull (on read)|Hybrid: push for normal, pull for celebs|
|Code exec isolation|Containers|MicroVM|Security vs startup speed|
|Counters|Async batch update|Sync DB write|Scale & hot-spot avoidance|
|Sharding|User_id hash|Time-based|Access pattern decides|
|Search|Elastic|DB LIKE|Relevance, scale|
|Realtime|WebSocket|Polling|Lower latency & bandwidth|

---

## F. Security & Observability Checklist

- **Security:** TLS everywhere, OAuth/JWT, rate limits, WAF, KMS encryption, audit logs.
    
- **Privacy:** GDPR delete, PII masking, access control.
    
- **Observability:** Metrics (SLOs), traces (OTel/Jaeger), logs (structured), alerts (SLO breaches).
    
- **Abuse/Fraud:** anomaly detection, CAPTCHAs, throttling.
    

---

## G. Common Interviewer Follow-ups

- “How will you **scale the database** when it grows?”
    
- “What if a **worker crashes mid-job**?”
    
- “How do you handle **hot keys / celebrity users**?”
    
- “What happens if **a region goes down**?” Multi-region DR plan.
    
- “How do you ensure **idempotency** on retries?”
    
- “How to **paginate & sort** efficiently?” (cursors > OFFSET).
    
- “How do you **invalidate caches** on updates?” (version keys, pub/sub).
    

Prepare 1–2 line responses for each.

---

## H. 15 Common System Design Questions (with ultra-short answer skeletons)

> Use the 10-step blueprint; below are the **core angles to highlight**.

1. **URL Shortener**
    
    - _Bottleneck:_ Read-heavy redirect.
        
    - _Key points:_ Base62 ID gen, cache redirects, SQL or KV store, TTL/custom aliases.
        
2. **Instagram/Twitter Feed**
    
    - _Bottleneck:_ Feed fanout & reads.
        
    - _Key points:_ Hybrid push/pull, Redis/Cassandra feed store, media on CDN.
        
3. **WhatsApp/Chat**
    
    - _Bottleneck:_ Real-time WS connections, ordering, durability.
        
    - _Key points:_ WS gateway, Cassandra for messages (partition by convo), receipts async, presence in Redis.
        
4. **Dropbox/Google Drive**
    
    - _Bottleneck:_ Sync & storage.
        
    - _Key points:_ Chunking + dedup, metadata SQL, object storage, sync via WS/long-poll, conflict resolution.
        
5. **Uber/Ride Hailing**
    
    - _Bottleneck:_ Geo index & matching latency.
        
    - _Key points:_ Geohash + Redis, Kafka for locations, trip FSM, payment idempotency.
        
6. **YouTube/Netflix**
    
    - _Bottleneck:_ Video streaming scale.
        
    - _Key points:_ Upload→transcode pipeline, HLS/DASH, CDN, async counters.
        
7. **LeetCode/Online Judge**
    
    - _Bottleneck:_ Secure code execution at scale.
        
    - _Key points:_ Queue + worker farm, Firecracker isolation, testcases in S3, P95 judge latency.
        
8. **Google Docs (Realtime Editor)**
    
    - _Bottleneck:_ Concurrent edits merging.
        
    - _Key points:_ OT or CRDT, WS sessions per doc, ops + snapshots, presence ephemeral.
        
9. **Rate Limiter (API Gateway)**
    
    - _Bottleneck:_ Fast counters per key.
        
    - _Key points:_ Token bucket/leaky bucket in Redis/Lua, distributed sync, sliding window.
        
10. **Notification/Email Service**
    
    - _Bottleneck:_ Fanout & retries.
        
    - _Key points:_ Topic-based queues, worker pools, exponential backoff, dedup & idempotency.
        
11. **Search Autocomplete**
    
    - _Bottleneck:_ Low-latency prefix queries.
        
    - _Key points:_ Trie or ES n-gram index, cache hot prefixes, debounce client.
        
12. **Analytics/Clickstream Pipeline**
    
    - _Bottleneck:_ High write, batch/stream processing.
        
    - _Key points:_ Ingest → Kafka → stream/batch (Flink/Spark) → DWH (BigQuery), cold storage.
        
13. **Ticket Booking (Seat Reservation)**
    
    - _Bottleneck:_ Concurrency & overbooking.
        
    - _Key points:_ Pessimistic locks or optimistic + retry, hold reservations with TTL, shards per event.
        
14. **Food Delivery (Swiggy/Zomato)**
    
    - _Bottleneck:_ Matching restaurants, delivery, real-time order state.
        
    - _Key points:_ Order svc, delivery svc, location tracking, status events, surge/ETA calc.
        
15. **Payment/Wallet Service**
    
    - _Bottleneck:_ Strong consistency & ledger accuracy.
        
    - _Key points:_ Double-entry ledger, idempotent operations, saga/outbox, PCI-compliant gateway.
        

---

## I. Flashcard Snippets

- **Idempotency token:** “client-generated UUID per request; server stores processed IDs”.
    
- **Cache invalidation:** “Versioned keys or pub/sub to evict; avoid stale reads.”
    
- **Hot partition fix:** “Consistent hashing w/ virtual nodes; split key space; add L2 cache.”
    
- **DLQ (dead-letter queue):** “Failed msgs > N retries go to DLQ for manual/auto replay.”
    
- **Backpressure:** “Consumer lag metric; scale consumers; rate-limit producers.”
    

---

## J. Last-Minute Tips

- Speak in **layers**: client → gateway → services → data.
    
- Always mention **metrics & alerts** at the end.
    
- Use **numbers** (even rough) to show realism.
    
- Show **trade-offs**: it’s not about the “perfect” design.
    
- Timebox: don’t deep dive early; breadth first, then drill when asked.
    

---
