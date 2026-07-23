# Semcode Shared Database Architecture & Usage Guide

`semcode` supports operating with a **Shared Central Database** shared across multiple repository trees, subsystem forks, and worktrees of the same project (such as Linux kernel trees: `linux-mainline`, `bpf-next`, `net-next`, `stable`).

This document explains why and how Shared Database mode works, how cross-repository data deduplication and branch isolation are maintained, and how to use it.

---

## 1. Why Use Shared Database Mode?

When working on large projects like the Linux kernel, developers frequently maintain multiple repository checkouts:
- Mainline (`linux-mainline`)
- Subsystem trees (`bpf-next`, `net-next`, `sound`)
- Stable release trees (`linux-6.12.y`, `linux-6.1.y`)
- Multiple local Git worktrees (`git worktree add ...`)

Because these trees share 90%+ of their Git commit graph and source code, indexing each tree into a separate database creates massive data duplication and wastes indexing time.

### Key Benefits of Shared Database Mode:
1. **Near-Instant Incremental Indexing Across Trees**: When you index a new subsystem tree or worktree into a shared database, `semcode` reuses all existing commit objects, file parsing results, and symbol AST definitions already indexed from other trees.
2. **Dramatic Storage Savings (50-80%)**: Content-addressed tables (functions, types, commit diffs, Blake3 content shards) are globally deduplicated across all repositories.
3. **Clean Branch Isolation**: Branch references (`indexed_branches`) are scoped per repository tree, preventing branch name collisions (e.g. `main` or `v6.12` existing in different trees).
4. **Git Worktree Consolidation**: Linked worktrees automatically map to their primary repository identity (`common_dir`), sharing branch references seamlessly without duplicate worktree records.

---

## 2. Core Architecture

### Global Content Deduplication vs. Relational Scoping

`semcode` divides its database tables into two distinct categories:

#### 1. Global Content Tables
* **Tables**: `functions`, `types`, `processed_files`, `git_commits`, `commit_vectors`, `content_0..15`
* **Strategy**: **Globally Deduplicated**
* **Details**: Keyed on SHA-1 file blob hashes, Blake3 content hashes, or Git commit SHAs. Identical function definitions or commit diffs across different repository trees are stored **once**.

#### 2. Scoped Identity Tables
* **Tables**: `indexed_branches`
* **Strategy**: **Repo-Scoped**
* **Details**: Stores `repo_path` (canonical repository identity path) to isolate branch references (`main`, `master`, `v6.12`) per repository checkout.

---

## 3. Database Locality Modes (`Local` vs `Shared`)

`semcode` automatically detects the database locality mode based on where the database file resides:

### 1. Local Database Mode (`.semcode.db`)
* **Trigger**: The database file is located inside a repository directory (e.g., `/home/user/linux/.semcode.db` or `/home/user/linux/.git/semcode.db`).
* **Behavior**:
  * Records `repo_path = NULL` or matches dynamically using `(repo_path = '{repo_identity}' OR repo_path IS NULL)`.
  * **100% Relocatable**: Moving or renaming the repository directory on disk will **never** invalidate database records or break future queries.

### 2. Shared Database Mode (`SEMCODE_DB`)
* **Trigger**: The database file resides in a central directory outside all repositories (e.g., `SEMCODE_DB=~/.cache/semcode.db` or `/var/shared/semcode.db`).
* **Behavior**:
  * Enforces strict SQL filtering on branch queries: `repo_path = '{repo_identity}'`.
  * Ensures complete isolation between different repository checkouts.

---

## 4. Git Worktree Identity Consolidation

Linked Git worktrees created via `git worktree add` share the same `.git` commit graph and refs as the parent repository. 

`semcode` automatically resolves worktrees to their parent repository's canonical identity (`common_dir`):
* Primary Checkout (`/home/user/linux`): Identity path `~/linux/.git`
* Linked Worktree (`/home/user/linux-feature-wt`): Identity path `~/linux/.git`

Because both resolve to `~/linux/.git`, worktrees transparently share the same branch index without creating redundant branch records.

---

## 5. Automatic Garbage Collection

In Shared Database mode, deleting a repository directory from the filesystem leaves behind stale `indexed_branches` records for that path.

`semcode-index` automatically invokes `DatabaseManager::garbage_collect()` at the end of every indexing run:
* **In Shared DB Mode**: Scans `indexed_branches` for `repo_path` entries whose paths no longer exist on the local filesystem and purges them via `table.delete()`.
* **In Local DB Mode**: Purges legacy unscoped `NULL` branch records if the database is running standalone outside of Git repositories.

---

## 6. How to Use Shared Database Mode

To use a shared central database across all your projects, set the `SEMCODE_DB` environment variable:

```bash
# 1. Index your mainline checkout into the central database
export SEMCODE_DB=~/.cache/semcode.db
cd ~/projects/linux-mainline
semcode-index -s .

# 2. Index a subsystem fork into the same central database (runs near-instantly!)
cd ~/projects/bpf-next
semcode-index -s .

# 3. Query within your current checkout (queries are automatically scoped to the active repository you are in)
cd ~/projects/bpf-next
semcode find_function bpf_prog_load
```

You can also add `export SEMCODE_DB=~/.cache/semcode.db` to your `~/.bashrc` or `~/.zshrc` to make all `semcode`, `semcode-index`, and MCP server instances automatically use the shared central database!
