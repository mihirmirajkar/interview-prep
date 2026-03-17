# Scenario 1: Document Processing Service — Code Review Exercise

## Context

You're joining a team that built an internal **Document Processing Service** in Python.
The service allows users to register, upload text documents, run analysis (word count,
sentiment, readability), search documents, and compare them for similarity.

A junior developer wrote most of the code and it's been running in a staging environment.
Your task is to **review this codebase**, identify issues, and prioritize fixes before
the service goes to production.

## Codebase Overview

| File            | Purpose                                      |
|-----------------|----------------------------------------------|
| `config.py`     | Application configuration and settings       |
| `models.py`     | Data models (User, Document, ProcessingResult) |
| `database.py`   | Database operations (SQLite)                 |
| `auth.py`       | Authentication — hashing, tokens, login      |
| `processor.py`  | Text processing and analysis logic           |
| `cache.py`      | In-memory caching layer                      |
| `utils.py`      | Utility/helper functions                     |
| `api.py`        | API request handler layer                    |

## Your Task

1. **Read through every file** (start wherever feels natural)
2. **Document every bug/issue** you find in `FINDINGS.md`
3. **Categorize** each by severity: CRITICAL / HIGH / MEDIUM / LOW
4. **Prioritize** — identify your top 3 fixes
5. Aim for **45–60 minutes** total

## What Interviewers Look For

- Can you spot **security vulnerabilities**?
- Do you catch **logic errors** that cause incorrect behavior?
- Do you notice **resource management** and **concurrency** issues?
- Do you understand **error handling** best practices?
- Can you distinguish critical from cosmetic issues?
- Are your proposed fixes correct and practical?

## Hints

- There are bugs in **every file**
- Severity ranges from "this will get us hacked" to "this is just sloppy"
- Some bugs are subtle — think about edge cases, error paths, and concurrent access
- There are **at least 15 issues** across the codebase

Good luck!
