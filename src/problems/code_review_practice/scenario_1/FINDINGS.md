# Code Review Findings

> **Instructions**: Review the codebase and document every issue you find below.
> Categorize each by severity: CRITICAL, HIGH, MEDIUM, or LOW.
> For each finding, note the file, line(s), description, and your proposed fix.
> Time yourself — aim for 45-60 minutes total.

---

## Severity Guide
- **CRITICAL**: Security vulnerabilities, data loss/corruption risks
- **HIGH**: Logic errors causing incorrect behavior, resource leaks, data integrity issues
- **MEDIUM**: Error handling gaps, reliability concerns, performance problems
- **LOW**: Code quality, style issues, maintainability concerns

---

## Findings

File"
api.py
1. ununsed imports - low
3. Handle_transforms is not using the payload to get_document and token is not verified - HIGH
4. Line 128 will fail if token expries - MEDIUM
5. payload can be NONE, in the whole file there is no check for that - HIGH - Fix priority number 2


auth.py
1. line 56 Uername and passord being printed in logs - CRITICAL - - Fix priority number 1
2. 67 Token logged - CRITICAL -- Fix priority number 1

cache.py
1. set does not use ttl - LOW
2. line 75 active entries can be different, because values come from two different dicts

config.py
1. db passowrd in plain text - CRITICAL - Fix priority number 1
2. line 38 exposes password - CRITICAL

---

## Answer Key — Full Bug List

### CRITICAL

**1. SQL Injection — `database.py` lines 68, 131, 138**
Three methods build SQL using f-strings with unsanitized user input:
```python
# Line 68 — get_user_by_username
query = f"SELECT * FROM users WHERE username = '{username}'"
# Line 131 — delete_document
f"DELETE FROM documents WHERE doc_id = {doc_id} AND owner_id = {owner_id}"
# Line 138 — search_documents
f"SELECT * FROM documents WHERE owner_id = {owner_id} AND (title LIKE '%{search_term}%'..."
```
**Impact**: Attacker can drop tables, exfiltrate all data, bypass authentication.
**Fix**: Use parameterized queries (`?` placeholders) like the other methods already do.

**2. Remote Code Execution via `eval()` — `processor.py` line 133**
```python
result = eval(transform_expr)
```
`apply_transform()` calls `eval()` on user-supplied input passed from `api.py` `handle_transform`.
**Impact**: Attacker can execute arbitrary Python: `__import__('os').system('rm -rf /')`.
**Fix**: Remove `eval()` entirely. Only allow the predefined safe transforms.

**3. Unsigned/Forgeable Auth Tokens — `auth.py` lines 30-45**
Tokens are just base64-encoded JSON with no HMAC or signature:
```python
token = base64.b64encode(json.dumps(payload).encode()).decode()
```
**Impact**: Anyone can forge a valid token for any user_id by base64-encoding their own JSON payload. The `SECRET_KEY` in config is never used.
**Fix**: Use HMAC signing (e.g., `hmac.new(SECRET_KEY, token_data, hashlib.sha256)`) or a proper JWT library.

**4. MD5 for Password Hashing — `auth.py` line 18**
```python
return hashlib.md5(password.encode()).hexdigest()
```
**Impact**: MD5 is unsalted, fast to brute-force, and has known collisions. Rainbow tables crack most MD5 hashes instantly.
**Fix**: Use `bcrypt`, `argon2`, or `hashlib.pbkdf2_hmac()` with a random salt.

**5. Credentials Logged in Plaintext — `auth.py` lines 56, 67** *(You caught this)*
```python
logger.info(f"Authentication attempt for user: {username} with password: {password}")
logger.info(f"Token generated for user {username}: {token}")
```
**Fix**: Never log passwords or tokens. Log only the username and event type.

**6. Hardcoded DB Password & SECRET_KEY — `config.py` lines 14, 20** *(You caught this)*
```python
DB_PASSWORD = "super_secret_password_123!"
SECRET_KEY = "a1b2c3d4e5f6789xyzinsecurekey"
```
**Fix**: Load from environment variables: `os.environ.get("DB_PASSWORD")`.

### HIGH

**7. Mutable Default Arguments — `models.py` lines 14, 29**
```python
permissions: list = []   # Line 14 — shared across ALL User instances
metadata: dict = {}      # Line 29 — shared across ALL Document instances
```
**Impact**: All users share the same permissions list. Adding a permission to one user adds it to everyone.
**Fix**: `permissions: list = field(default_factory=list)` and `metadata: dict = field(default_factory=dict)`.

**8. No Auth Check on payload=None — `api.py` multiple methods** *(You caught this)*
`verify_token()` returns `None` on expired/invalid tokens, but `handle_upload`, `handle_process`, `handle_transform`, `handle_search`, `handle_compare`, `handle_delete`, `handle_list` all use `payload["user_id"]` without checking for `None`.
**Impact**: Every authenticated endpoint crashes (or worse, bypasses auth) on invalid tokens.
**Fix**: Add `if not payload: return {"status": "error", "message": "Authentication required"}` at the top of each handler.

**9. Off-by-One Pagination — `database.py` line 119**
```python
offset = page * page_size  # page=1 → offset=20, skips all of page 1
```
**Impact**: First page returns zero results; all pages are shifted by one.
**Fix**: `offset = (page - 1) * page_size`.

**10. Resource Leak — `processor.py` lines 100-101**
```python
f = open(filepath, 'r', encoding='utf-8')
content = f.read()
# f is never closed — especially if the length check returns early
```
**Fix**: Use `with open(filepath, 'r', encoding='utf-8') as f:`.

**11. `handle_compare` — No Null/Ownership Check — `api.py` lines 130-135**
If either document doesn't exist, `doc1["content"]` raises `TypeError: 'NoneType' is not subscriptable`. Also no check that the user owns these documents.
**Fix**: Check both docs for `None` and verify `owner_id` matches.

**12. `DocumentCollection.remove()` Doesn't Clean Index — `models.py` line 73**
Documents are deleted from `_documents` but their word entries remain in `_index`, causing search to return ghost doc_ids that crash on lookup.
**Fix**: Remove the document's words from `_index` before deleting from `_documents`.

### MEDIUM

**13. Unreachable Except Clause — `database.py` lines 148-150**
```python
except Exception:
    return 0
except sqlite3.OperationalError as e:   # DEAD CODE — never reached
    logger.error(f"Database error: {e}")
    return -1
```
**Impact**: Database errors are silently swallowed; the more specific handler never runs.
**Fix**: Swap the order — put `sqlite3.OperationalError` first, or remove it.

**14. Cache Not Thread-Safe — `cache.py`**
`_lock = threading.Lock()` is defined but never acquired in `get()`, `set()`, or `delete()`. Under concurrent access, dictionary operations can corrupt state.
**Fix**: Wrap cache operations in `with _lock:`.

**15. `handle_transform` — No Ownership Check — `api.py` line 114** *(You caught this)*
Any authenticated user can transform any document — no check that `doc["owner_id"] == payload["user_id"]`.

**16. Bare `except:` Clauses — `api.py` lines 166, 178**
```python
except:
    results.append({"status": "error", "doc_id": doc_id})
```
**Impact**: Catches `KeyboardInterrupt`, `SystemExit`, `MemoryError`, etc.
**Fix**: Use `except Exception:`.

**17. `cache.set()` Ignores TTL Parameter — `cache.py` line 34** *(You caught this)*
The `ttl` argument is accepted but never used; all entries use `DEFAULT_TTL`.

### LOW

**18. `DEBUG = True` in Production Config — `config.py` line 8**
Debug mode left enabled, which typically exposes stack traces and sensitive data.

**19. Weak Password Policy — `auth.py` line 74**
Minimum password length is only 4 characters, no complexity requirements.

**20. `cache.set` Shadows Built-in — `cache.py` line 31**
Function named `set` shadows Python's built-in `set` type within this module.

**21. No Email Validation on Register — `api.py` `handle_register`**
`validate_email` is imported from utils but never called. Users can register with garbage emails.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 6     |
| HIGH     | 6     |
| MEDIUM   | 5     |
| LOW      | 4     |
| **Total**| **21**|

## Top 3 Priority Fixes

1. **SQL Injection** (`database.py`) — trivially exploitable, full DB compromise
2. **`eval()` RCE** (`processor.py`) — arbitrary code execution from user input
3. **Unsigned tokens** (`auth.py`) — anyone can forge auth as any user
