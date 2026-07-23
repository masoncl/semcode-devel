// SPDX-License-Identifier: MIT OR Apache-2.0
//! Store for tracking indexed git branches.
//!
//! This module provides functionality to track which branches have been indexed,
//! their tip commits, and when they were last indexed. This enables efficient
//! multi-branch indexing by avoiding re-indexing branches that haven't changed.

use anyhow::Result;
use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::index::{scalar::BTreeIndexBuilder, Index as LanceIndex};
use std::sync::Arc;

/// Information about an indexed branch
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedBranchInfo {
    /// Branch name (e.g., "main", "origin/develop")
    pub branch_name: String,
    /// The commit SHA at the tip of the branch when indexed
    pub tip_commit: String,
    /// Unix timestamp of when the branch was last indexed
    pub indexed_at: i64,
    /// Remote name if this is a remote-tracking branch (e.g., "origin")
    pub remote: Option<String>,
    /// Repository canonical path identity (NULL for local DB, canonical path for shared DB)
    pub repo_path: Option<String>,
}

/// JSON-serializable version of IndexedBranchInfo
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexedBranchInfoJson {
    pub branch_name: String,
    pub tip_commit: String,
    pub indexed_at: i64,
    pub remote: Option<String>,
    pub repo_path: Option<String>,
}

impl From<IndexedBranchInfo> for IndexedBranchInfoJson {
    fn from(info: IndexedBranchInfo) -> Self {
        IndexedBranchInfoJson {
            branch_name: info.branch_name,
            tip_commit: info.tip_commit,
            indexed_at: info.indexed_at,
            remote: info.remote,
            repo_path: info.repo_path,
        }
    }
}

impl From<IndexedBranchInfoJson> for IndexedBranchInfo {
    fn from(json: IndexedBranchInfoJson) -> Self {
        IndexedBranchInfo {
            branch_name: json.branch_name,
            tip_commit: json.tip_commit,
            indexed_at: json.indexed_at,
            remote: json.remote,
            repo_path: json.repo_path,
        }
    }
}

/// Store for managing indexed branch records
pub struct IndexedBranchStore {
    connection: Connection,
    repo_identity: String,
    is_local_db: bool,
}

impl IndexedBranchStore {
    pub fn new(connection: Connection, repo_identity: String, is_local_db: bool) -> Self {
        Self {
            connection,
            repo_identity,
            is_local_db,
        }
    }

    /// Construct SQL filter string for LanceDB queries based on repository identity and database locality.
    /// Returns "true" (noop) if the table schema on disk does not have a repo_path column (legacy schema).
    pub fn build_sql_filter(&self, schema: &Schema) -> String {
        if schema.column_with_name("repo_path").is_none() {
            return "true".to_string();
        }
        let escaped_rp = Self::escape_sql_literal(&self.repo_identity);
        if self.is_local_db {
            format!("(repo_path = '{escaped_rp}' OR repo_path IS NULL)")
        } else {
            format!("repo_path = '{escaped_rp}'")
        }
    }

    async fn open_table_with_schema(&self) -> Result<(lancedb::Table, Arc<Schema>)> {
        let table = self
            .connection
            .open_table("indexed_branches")
            .execute()
            .await?;
        let schema = table.schema().await?;
        Ok((table, schema))
    }

    async fn add_repo_path_column(&self, table: &lancedb::Table) -> Result<()> {
        tracing::info!("Adding missing 'repo_path' column to indexed_branches table during write");
        table
            .add_columns(
                lancedb::table::NewColumnTransform::SqlExpressions(vec![(
                    "repo_path".into(),
                    "CAST(NULL AS string)".into(),
                )]),
                None,
            )
            .await?;

        let _ = table
            .create_index(
                &["repo_path"],
                LanceIndex::BTree(BTreeIndexBuilder::default()),
            )
            .execute()
            .await;

        Ok(())
    }

    /// Get the Arrow schema for the indexed_branches table
    pub fn get_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("branch_name", DataType::Utf8, false),
            Field::new("tip_commit", DataType::Utf8, false),
            Field::new("indexed_at", DataType::Int64, false),
            Field::new("remote", DataType::Utf8, true),
            Field::new("repo_path", DataType::Utf8, true),
        ]))
    }

    fn escape_sql_literal(val: &str) -> String {
        val.replace("'", "''")
    }

    /// Record that a branch has been indexed at a specific commit
    pub async fn record_branch_indexed(&self, info: &IndexedBranchInfo) -> Result<()> {
        // First, remove any existing record for this branch under current repo scoping
        self.remove_branch(&info.branch_name).await?;

        let (table, schema) = self.open_table_with_schema().await?;

        // Migrate the schema to support scoping if writing to an older version
        if schema.column_with_name("repo_path").is_none() {
            self.add_repo_path_column(&table).await?;
        }

        // Build arrays for each column
        let mut branch_name_builder = StringBuilder::new();
        let mut tip_commit_builder = StringBuilder::new();
        let mut indexed_at_builder = arrow::array::Int64Builder::new();
        let mut remote_builder = StringBuilder::new();
        let mut repo_path_builder = StringBuilder::new();

        branch_name_builder.append_value(&info.branch_name);
        tip_commit_builder.append_value(&info.tip_commit);
        indexed_at_builder.append_value(info.indexed_at);
        match &info.remote {
            Some(r) => remote_builder.append_value(r),
            None => remote_builder.append_null(),
        }
        match &info.repo_path {
            Some(rp) => repo_path_builder.append_value(rp),
            None => repo_path_builder.append_null(),
        }

        let batch = RecordBatch::try_from_iter(vec![
            (
                "branch_name",
                Arc::new(branch_name_builder.finish()) as ArrayRef,
            ),
            (
                "tip_commit",
                Arc::new(tip_commit_builder.finish()) as ArrayRef,
            ),
            (
                "indexed_at",
                Arc::new(indexed_at_builder.finish()) as ArrayRef,
            ),
            ("remote", Arc::new(remote_builder.finish()) as ArrayRef),
            (
                "repo_path",
                Arc::new(repo_path_builder.finish()) as ArrayRef,
            ),
        ])?;

        table.add(vec![batch]).execute().await?;

        Ok(())
    }

    /// Get the tip commit for a specific branch
    pub async fn get_branch_tip(&self, branch_name: &str) -> Result<Option<String>> {
        let info = self.get_branch_info(branch_name).await?;
        Ok(info.map(|i| i.tip_commit))
    }

    /// Get full information about a specific branch
    pub async fn get_branch_info(&self, branch_name: &str) -> Result<Option<IndexedBranchInfo>> {
        let (table, schema) = self.open_table_with_schema().await?;

        let escaped_name = Self::escape_sql_literal(branch_name);
        let repo_filter = self.build_sql_filter(&schema);
        let sql_filter = format!("{repo_filter} AND branch_name = '{escaped_name}'");

        let results = table
            .query()
            .only_if(sql_filter)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        if results.is_empty() {
            return Ok(None);
        }

        // Disambiguate between legacy un-scoped (repo_path = NULL) and repo-scoped records
        // by selecting the record with the most recent indexed_at timestamp.
        let newest = results
            .iter()
            .flat_map(|batch| (0..batch.num_rows()).map(move |i| (batch, i)))
            .map(|(batch, i)| self.extract_record_from_batch(batch, i))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .max_by_key(|info| info.indexed_at);

        Ok(newest)
    }

    /// List all indexed branches
    pub async fn list_indexed_branches(&self) -> Result<Vec<IndexedBranchInfo>> {
        let (table, schema) = self.open_table_with_schema().await?;

        let sql_filter = self.build_sql_filter(&schema);
        let query = table.query().only_if(sql_filter);

        let results = query.execute().await?.try_collect::<Vec<_>>().await?;

        let mut branches = Vec::new();
        for batch in &results {
            for i in 0..batch.num_rows() {
                if let Some(info) = self.extract_record_from_batch(batch, i)? {
                    branches.push(info);
                }
            }
        }

        // Sort by branch name for consistent output
        branches.sort_by(|a, b| a.branch_name.cmp(&b.branch_name));

        Ok(branches)
    }

    /// Check if a branch is indexed at the current tip commit
    pub async fn is_branch_current(&self, branch_name: &str, current_tip: &str) -> Result<bool> {
        if let Some(info) = self.get_branch_info(branch_name).await? {
            Ok(info.tip_commit == current_tip)
        } else {
            Ok(false)
        }
    }

    /// Remove a branch record (used when branch is deleted or before updating)
    pub async fn remove_branch(&self, branch_name: &str) -> Result<()> {
        let (table, schema) = self.open_table_with_schema().await?;

        let escaped_name = Self::escape_sql_literal(branch_name);
        let repo_filter = self.build_sql_filter(&schema);
        let sql_filter = format!("{repo_filter} AND branch_name = '{escaped_name}'");

        table.delete(&sql_filter).await?;
        Ok(())
    }

    /// Remove all branches that match a remote prefix (e.g., "origin/")
    pub async fn remove_branches_by_remote(&self, remote: &str) -> Result<usize> {
        let branches = self.list_indexed_branches().await?;
        let mut removed = 0;

        for branch in branches {
            if branch.remote.as_deref() == Some(remote) {
                self.remove_branch(&branch.branch_name).await?;
                removed += 1;
            }
        }

        Ok(removed)
    }

    /// Get all branches that point to a specific commit
    pub async fn get_branches_at_commit(&self, commit_sha: &str) -> Result<Vec<IndexedBranchInfo>> {
        let (table, schema) = self.open_table_with_schema().await?;

        let escaped_sha = Self::escape_sql_literal(commit_sha);
        let repo_filter = self.build_sql_filter(&schema);
        let sql_filter = format!("{repo_filter} AND tip_commit = '{escaped_sha}'");

        let results = table
            .query()
            .only_if(sql_filter)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut branches = Vec::new();
        for batch in &results {
            for i in 0..batch.num_rows() {
                if let Some(info) = self.extract_record_from_batch(batch, i)? {
                    branches.push(info);
                }
            }
        }

        Ok(branches)
    }

    /// Get total count of indexed branches matching repository filter
    pub async fn count(&self) -> Result<usize> {
        let branches = self.list_indexed_branches().await?;
        Ok(branches.len())
    }

    /// Extract an IndexedBranchInfo from a batch at the given row index
    pub fn extract_record_from_batch(
        &self,
        batch: &RecordBatch,
        row: usize,
    ) -> Result<Option<IndexedBranchInfo>> {
        let branch_name_array = batch
            .column_by_name("branch_name")
            .ok_or_else(|| anyhow::anyhow!("Missing branch_name column"))?
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Invalid branch_name column type"))?;

        let tip_commit_array = batch
            .column_by_name("tip_commit")
            .ok_or_else(|| anyhow::anyhow!("Missing tip_commit column"))?
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Invalid tip_commit column type"))?;

        let indexed_at_array = batch
            .column_by_name("indexed_at")
            .ok_or_else(|| anyhow::anyhow!("Missing indexed_at column"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("Invalid indexed_at column type"))?;

        let remote_array = batch
            .column_by_name("remote")
            .ok_or_else(|| anyhow::anyhow!("Missing remote column"))?
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Invalid remote column type"))?;

        // Extract optional repo_path if column exists and value at row is non-null
        let repo_path = batch
            .column_by_name("repo_path")
            .and_then(|col| col.as_any().downcast_ref::<arrow::array::StringArray>())
            .filter(|arr| !arr.is_null(row))
            .map(|arr| arr.value(row).to_string());

        let branch_name = branch_name_array.value(row).to_string();
        let tip_commit = tip_commit_array.value(row).to_string();
        let indexed_at = indexed_at_array.value(row);
        let remote = if remote_array.is_null(row) {
            None
        } else {
            Some(remote_array.value(row).to_string())
        };

        Ok(Some(IndexedBranchInfo {
            branch_name,
            tip_commit,
            indexed_at,
            remote,
            repo_path,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_store() -> (TempDir, IndexedBranchStore) {
        let tmpdir = TempDir::new().unwrap();
        let db_path = tmpdir.path().to_str().unwrap();
        let connection = lancedb::connect(db_path).execute().await.unwrap();

        // Create the table
        let schema = IndexedBranchStore::get_schema();
        let empty_batch = RecordBatch::new_empty(schema.clone());
        connection
            .create_table("indexed_branches", vec![empty_batch])
            .execute()
            .await
            .unwrap();

        let store = IndexedBranchStore::new(connection, "~/projects/repoA/.git".to_string(), true);
        (tmpdir, store)
    }

    #[tokio::test]
    async fn test_record_and_get_branch() {
        let (_tmpdir, store) = create_test_store().await;

        let info = IndexedBranchInfo {
            branch_name: "main".to_string(),
            tip_commit: "abc123def456".to_string(),
            indexed_at: 1699900000,
            remote: None,
            repo_path: None,
        };

        store.record_branch_indexed(&info).await.unwrap();

        let retrieved = store.get_branch_info("main").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.branch_name, "main");
        assert_eq!(retrieved.tip_commit, "abc123def456");
        assert_eq!(retrieved.indexed_at, 1699900000);
        assert!(retrieved.remote.is_none());
    }

    #[tokio::test]
    async fn test_record_remote_branch() {
        let (_tmpdir, store) = create_test_store().await;

        let info = IndexedBranchInfo {
            branch_name: "origin/develop".to_string(),
            tip_commit: "789abc123".to_string(),
            indexed_at: 1699900100,
            remote: Some("origin".to_string()),
            repo_path: None,
        };

        store.record_branch_indexed(&info).await.unwrap();

        let retrieved = store.get_branch_info("origin/develop").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.remote, Some("origin".to_string()));
    }

    #[tokio::test]
    async fn test_update_branch() {
        let (_tmpdir, store) = create_test_store().await;

        // Record initial version
        let info1 = IndexedBranchInfo {
            branch_name: "main".to_string(),
            tip_commit: "commit1".to_string(),
            indexed_at: 1699900000,
            remote: None,
            repo_path: None,
        };
        store.record_branch_indexed(&info1).await.unwrap();

        // Update with new commit
        let info2 = IndexedBranchInfo {
            branch_name: "main".to_string(),
            tip_commit: "commit2".to_string(),
            indexed_at: 1699900100,
            remote: None,
            repo_path: None,
        };
        store.record_branch_indexed(&info2).await.unwrap();

        // Should have only one record with the new commit
        let retrieved = store.get_branch_info("main").await.unwrap().unwrap();
        assert_eq!(retrieved.tip_commit, "commit2");
        assert_eq!(retrieved.indexed_at, 1699900100);

        // Count should be 1
        assert_eq!(store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_list_indexed_branches() {
        let (_tmpdir, store) = create_test_store().await;

        let branches = vec![
            IndexedBranchInfo {
                branch_name: "main".to_string(),
                tip_commit: "commit1".to_string(),
                indexed_at: 1699900000,
                remote: None,
                repo_path: None,
            },
            IndexedBranchInfo {
                branch_name: "develop".to_string(),
                tip_commit: "commit2".to_string(),
                indexed_at: 1699900100,
                remote: None,
                repo_path: None,
            },
            IndexedBranchInfo {
                branch_name: "origin/feature".to_string(),
                tip_commit: "commit3".to_string(),
                indexed_at: 1699900200,
                remote: Some("origin".to_string()),
                repo_path: None,
            },
        ];

        for info in &branches {
            store.record_branch_indexed(info).await.unwrap();
        }

        let listed = store.list_indexed_branches().await.unwrap();
        assert_eq!(listed.len(), 3);

        // Should be sorted by name
        assert_eq!(listed[0].branch_name, "develop");
        assert_eq!(listed[1].branch_name, "main");
        assert_eq!(listed[2].branch_name, "origin/feature");
    }

    #[tokio::test]
    async fn test_is_branch_current() {
        let (_tmpdir, store) = create_test_store().await;

        let info = IndexedBranchInfo {
            branch_name: "main".to_string(),
            tip_commit: "abc123".to_string(),
            indexed_at: 1699900000,
            remote: None,
            repo_path: None,
        };
        store.record_branch_indexed(&info).await.unwrap();

        assert!(store.is_branch_current("main", "abc123").await.unwrap());
        assert!(!store.is_branch_current("main", "def456").await.unwrap());
        assert!(!store
            .is_branch_current("nonexistent", "abc123")
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_remove_branch() {
        let (_tmpdir, store) = create_test_store().await;

        let info = IndexedBranchInfo {
            branch_name: "feature".to_string(),
            tip_commit: "abc123".to_string(),
            indexed_at: 1699900000,
            remote: None,
            repo_path: None,
        };
        store.record_branch_indexed(&info).await.unwrap();

        assert!(store.get_branch_info("feature").await.unwrap().is_some());

        store.remove_branch("feature").await.unwrap();

        assert!(store.get_branch_info("feature").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_remove_branches_by_remote() {
        let (_tmpdir, store) = create_test_store().await;

        let branches = vec![
            IndexedBranchInfo {
                branch_name: "main".to_string(),
                tip_commit: "commit1".to_string(),
                indexed_at: 1699900000,
                remote: None,
                repo_path: None,
            },
            IndexedBranchInfo {
                branch_name: "origin/main".to_string(),
                tip_commit: "commit2".to_string(),
                indexed_at: 1699900100,
                remote: Some("origin".to_string()),
                repo_path: None,
            },
            IndexedBranchInfo {
                branch_name: "origin/develop".to_string(),
                tip_commit: "commit3".to_string(),
                indexed_at: 1699900200,
                remote: Some("origin".to_string()),
                repo_path: None,
            },
            IndexedBranchInfo {
                branch_name: "upstream/main".to_string(),
                tip_commit: "commit4".to_string(),
                indexed_at: 1699900300,
                remote: Some("upstream".to_string()),
                repo_path: None,
            },
        ];

        for info in &branches {
            store.record_branch_indexed(info).await.unwrap();
        }

        let removed = store.remove_branches_by_remote("origin").await.unwrap();
        assert_eq!(removed, 2);

        let remaining = store.list_indexed_branches().await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.iter().any(|b| b.branch_name == "main"));
        assert!(remaining.iter().any(|b| b.branch_name == "upstream/main"));
    }

    #[tokio::test]
    async fn test_get_branches_at_commit() {
        let (_tmpdir, store) = create_test_store().await;

        let shared_commit = "shared123";
        let branches = vec![
            IndexedBranchInfo {
                branch_name: "main".to_string(),
                tip_commit: shared_commit.to_string(),
                indexed_at: 1699900000,
                remote: None,
                repo_path: None,
            },
            IndexedBranchInfo {
                branch_name: "release".to_string(),
                tip_commit: shared_commit.to_string(),
                indexed_at: 1699900100,
                remote: None,
                repo_path: None,
            },
            IndexedBranchInfo {
                branch_name: "develop".to_string(),
                tip_commit: "different456".to_string(),
                indexed_at: 1699900200,
                remote: None,
                repo_path: None,
            },
        ];

        for info in &branches {
            store.record_branch_indexed(info).await.unwrap();
        }

        let at_shared = store.get_branches_at_commit(shared_commit).await.unwrap();
        assert_eq!(at_shared.len(), 2);
        assert!(at_shared.iter().any(|b| b.branch_name == "main"));
        assert!(at_shared.iter().any(|b| b.branch_name == "release"));
    }

    #[tokio::test]
    async fn test_get_branch_tip() {
        let (_tmpdir, store) = create_test_store().await;

        let info = IndexedBranchInfo {
            branch_name: "main".to_string(),
            tip_commit: "abc123".to_string(),
            indexed_at: 1699900000,
            remote: None,
            repo_path: None,
        };
        store.record_branch_indexed(&info).await.unwrap();

        assert_eq!(
            store.get_branch_tip("main").await.unwrap(),
            Some("abc123".to_string())
        );
        assert_eq!(store.get_branch_tip("nonexistent").await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_branch_name_with_special_chars() {
        let (_tmpdir, store) = create_test_store().await;

        // Test branch names with characters that need escaping
        let info = IndexedBranchInfo {
            branch_name: "feature/user's-branch".to_string(),
            tip_commit: "abc123".to_string(),
            indexed_at: 1699900000,
            remote: None,
            repo_path: None,
        };
        store.record_branch_indexed(&info).await.unwrap();

        let retrieved = store
            .get_branch_info("feature/user's-branch")
            .await
            .unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().branch_name, "feature/user's-branch");
    }

    #[tokio::test]
    async fn test_shared_db_branch_isolation() {
        let tmpdir = tempfile::tempdir().unwrap();
        let connection = lancedb::connect(tmpdir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let schema = IndexedBranchStore::get_schema();
        let empty_batch = RecordBatch::new_empty(schema.clone());
        connection
            .create_table("indexed_branches", vec![empty_batch])
            .execute()
            .await
            .unwrap();

        let repo_a = "~/projects/repoA/.git";
        let repo_b = "~/projects/repoB/.git";

        let store_a = IndexedBranchStore::new(connection.clone(), repo_a.to_string(), false);
        let store_b = IndexedBranchStore::new(connection.clone(), repo_b.to_string(), false);

        let info_a = IndexedBranchInfo {
            branch_name: "main".to_string(),
            tip_commit: "commit_a_123".to_string(),
            indexed_at: 1699900000,
            remote: None,
            repo_path: Some(repo_a.to_string()),
        };
        store_a.record_branch_indexed(&info_a).await.unwrap();

        let info_b = IndexedBranchInfo {
            branch_name: "main".to_string(),
            tip_commit: "commit_b_456".to_string(),
            indexed_at: 1699900100,
            remote: None,
            repo_path: Some(repo_b.to_string()),
        };
        store_b.record_branch_indexed(&info_b).await.unwrap();

        // In shared mode, store A sees only repo A's main branch
        let branch_a = store_a.get_branch_info("main").await.unwrap().unwrap();
        assert_eq!(branch_a.tip_commit, "commit_a_123");

        // Store B sees only repo B's main branch
        let branch_b = store_b.get_branch_info("main").await.unwrap().unwrap();
        assert_eq!(branch_b.tip_commit, "commit_b_456");

        // Listing for Store A yields 1 branch
        let list_a = store_a.list_indexed_branches().await.unwrap();
        assert_eq!(list_a.len(), 1);
        assert_eq!(list_a[0].tip_commit, "commit_a_123");
    }

    #[tokio::test]
    async fn test_local_db_matches_null_and_active() {
        let tmpdir = tempfile::tempdir().unwrap();
        let connection = lancedb::connect(tmpdir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let schema = IndexedBranchStore::get_schema();
        let empty_batch = RecordBatch::new_empty(schema.clone());
        connection
            .create_table("indexed_branches", vec![empty_batch])
            .execute()
            .await
            .unwrap();

        let repo_a = "~/projects/repoA/.git";

        let store_local = IndexedBranchStore::new(connection.clone(), repo_a.to_string(), true);
        let store_shared = IndexedBranchStore::new(connection.clone(), repo_a.to_string(), false);

        // Record a legacy NULL branch entry
        let info_null = IndexedBranchInfo {
            branch_name: "legacy_branch".to_string(),
            tip_commit: "legacy_sha".to_string(),
            indexed_at: 1699900000,
            remote: None,
            repo_path: None,
        };
        store_local.record_branch_indexed(&info_null).await.unwrap();

        // In Local mode, the legacy NULL branch is returned
        let branch = store_local
            .get_branch_info("legacy_branch")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(branch.tip_commit, "legacy_sha");

        // In Shared mode, the legacy NULL branch is NOT returned
        let branch_shared = store_shared.get_branch_info("legacy_branch").await.unwrap();
        assert!(branch_shared.is_none());
    }

    #[tokio::test]
    async fn test_read_old_indexed_branches_table_does_not_alter_schema() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().to_str().unwrap();

        // 1. Manually create an old 4-column indexed_branches table FIRST
        let connection = lancedb::connect(db_path).execute().await.unwrap();
        let old_schema = Arc::new(Schema::new(vec![
            Field::new("branch_name", DataType::Utf8, false),
            Field::new("tip_commit", DataType::Utf8, false),
            Field::new("indexed_at", DataType::Int64, false),
            Field::new("remote", DataType::Utf8, true),
        ]));

        let empty_batch = RecordBatch::new_empty(old_schema);
        connection
            .create_table("indexed_branches", vec![empty_batch])
            .execute()
            .await
            .unwrap();

        // 2. NOW open IndexedBranchStore on the old schema
        let store = IndexedBranchStore::new(connection, "~/projects/repoA/.git".to_string(), true);

        // Reading branches works cleanly without altering disk schema
        let (_table, schema_read) = store.open_table_with_schema().await.unwrap();
        assert!(
            schema_read.column_with_name("repo_path").is_none(),
            "Read-only operations should not alter schema on disk"
        );

        let branches = store.list_indexed_branches().await.unwrap();
        assert!(branches.is_empty());

        // Writing a new branch record triggers schema migration on demand
        let info = IndexedBranchInfo {
            branch_name: "main".to_string(),
            tip_commit: "sha123".to_string(),
            indexed_at: 1699900000,
            remote: None,
            repo_path: None,
        };
        store.record_branch_indexed(&info).await.unwrap();

        let (_table, schema_after_write) = store.open_table_with_schema().await.unwrap();
        assert!(
            schema_after_write.column_with_name("repo_path").is_some(),
            "Write operation should add repo_path column on demand"
        );
    }
}
