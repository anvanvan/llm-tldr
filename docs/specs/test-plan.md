## Test Plan

### Summary
- Total tests written: 10 (7 RED + 3 pre-passing regression guards)
- Test files created:
  - `tests/test_g6_g3_g10_guards.py`
  - `tests/test_incr_full_equivalence.py`
  - `tests/test_s5_no_rehash.py`
- All RED tests verified FAILING: YES
- All guard tests verified PASSING: YES
- Runner command used: `python3 -m pytest --no-cov`

### Test Inventory

| # | Test File | Test Name | Expected Failure | Actual Failure (3 lines max) | Status |
|---|-----------|-----------|-----------------|------------------------------|--------|
| 1 | tests/test_g6_g3_g10_guards.py | TestG6FallThroughGuard::test_out_of_scope_hint_falls_through_to_hash_floor | file_a.py not re-parsed when out-of-scope hint normalizes to empty | AssertionError: G-6 FALL-THROUGH FAIL: file_a.py was edited but NOT re-parsed after reindex with an out-of-scope hint. Paths processed: []. assert False | RED ✓ |
| 2 | tests/test_g6_g3_g10_guards.py | TestG3GitignoreParity::test_gitignored_file_absent_from_live_set | generated.py (gitignored) present in live set | AssertionError: G-3 FAIL: `generated.py` is gitignored but appears in the live set. Live files: ['generated.py', 'module.py']. assert 'generated.py' not in {'generated.py', 'module.py'} | RED ✓ |
| 3 | tests/test_g6_g3_g10_guards.py | TestG3GitignoreParity::test_false_dirty_caused_by_gitignored_file | generated.py still in live set after cold build | AssertionError: G-3 FAIL: generated.py (gitignored) is still in the live set after cold build. Live set: ['generated.py', 'module.py']. assert 'generated.py' not in {'generated.py', 'module.py'} | RED ✓ |
| 4 | tests/test_incr_full_equivalence.py | TestIncrementalEqualsFullMultiFile::test_per_unit_calls_and_called_by_exact_equality | Pass-2b bare-name key pollution causes called_by divergence | AssertionError: Incremental call-graph diverges from --full rebuild (3 problem(s)): db_insert.called_by: incremental=['User::save', 'create_user', 'save'] != full=['create_user', 'save']. assert not [...] | RED ✓ |
| 5 | tests/test_incr_full_equivalence.py | TestIncrementalEqualsFullMultiFile::test_incremental_reembeds_only_new_file_not_all_units | Re-embed count 7 > 5 due to Pass-2b pollution | AssertionError: Re-embed count (7) exceeds 5 after adding reports.py. assert 7 <= 5. RED: Pass-2b pollution adds bare-name duplicate caller entries | RED ✓ |
| 6 | tests/test_s5_no_rehash.py | TestS5NoRehashOnNoOp::test_noop_reindex_does_not_rehash_unchanged_files | compute_file_hash called 2 times on no-op (should be 0) | AssertionError: S-5 FAIL: compute_file_hash must NOT be called on a no-op reindex. Call count: 2 (expected 0). assert 2 == 0 | RED ✓ |
| 7 | tests/test_s5_no_rehash.py | TestS5NoRehashOnNoOp::test_incremental_reindex_does_not_rehash_unchanged_files | compute_file_hash called 3 times on incremental (should be 0) | AssertionError: S-5 FAIL: after editing beta.py, the incremental reindex persist must NOT call compute_file_hash. Call count: 3 (expected 0). assert 3 == 0 | RED ✓ |
| 8 | tests/test_g6_g3_g10_guards.py | TestG6FallThroughGuard::test_in_scope_hint_still_processes_changed_file | (pre-passing guard) | PASSES on HEAD | GUARD ✓ |
| 9 | tests/test_g6_g3_g10_guards.py | TestG10SubdirSnapshotKeyConsistency::test_subdir_reindex_reuses_unchanged_file | (pre-passing guard) | PASSES on HEAD | GUARD ✓ |
| 10 | tests/test_g6_g3_g10_guards.py | TestG10SubdirSnapshotKeyConsistency::test_subdir_noop_reindex_reuses_all_files | (pre-passing guard) | PASSES on HEAD | GUARD ✓ |

### Anti-Pattern Check
- [x] No tests for implementation details (testing behavior only)
- [x] No mocks for non-existent code
- [x] No tests that pass immediately (all 7 RED tests confirmed failing)
- [x] All failures are for the right reason (missing feature logic, not broken test)
- [x] Test names read as behavior specifications
- [x] Tests use project's existing patterns and utilities (_make_fake_model, _build_two_file_repo, patch("tldr.semantic.get_model"), monkeypatch.setenv("TLDR_MAX_WORKERS", "1"))

### RED Evidence

#### tests/test_g6_g3_g10_guards.py
```
AssertionError: G-6 FALL-THROUGH FAIL: file_a.py was edited but NOT re-parsed after reindex with an out-of-scope hint. The hint normalized to changed=set(); without the fall-through guard, branch (a) carries all files and skips the hash floor. Paths processed: [].
AssertionError: G-3 FAIL: `generated.py` is gitignored but appears in the live set. Live files: ['generated.py', 'module.py']. _enumerate_live_files must exclude gitignored files exactly as extract_units_from_project does via should_ignore.
AssertionError: G-3 FAIL: generated.py (gitignored) is still in the live set after cold build. Live set: ['generated.py', 'module.py'].
```

#### tests/test_incr_full_equivalence.py
```
AssertionError: Incremental call-graph diverges from --full rebuild (3 problem(s)):
  db_insert.called_by: incremental=['User::save', 'create_user', 'save'] != full=['create_user', 'save']
  db_delete.called_by: incremental=['Admin::delete', 'delete', 'generate_report'] != full=['delete', 'generate_report']
AssertionError: Re-embed count (7) exceeds 5 after adding reports.py (1 new file with 1 function). assert 7 <= 5.
```

#### tests/test_s5_no_rehash.py
```
AssertionError: S-5 FAIL: compute_file_hash must NOT be called on a no-op reindex (all files unchanged → all in old_snapshot → sha1 reused from case-2). Call count: 2 (expected 0). assert 2 == 0
AssertionError: S-5 FAIL: after editing beta.py, the incremental reindex persist must NOT call compute_file_hash for any file. Call count: 3 (expected 0). assert 3 == 0
```

### Full Details

**Features targeted:**

1. **G-6 Fall-through guard** (`test_g6_g3_g10_guards.py`): When `dirty_files` hint is non-empty but ALL entries normalize to out-of-scope paths, `_derive_dirty_set` branch (a) currently returns `(set(), deleted)` without falling through to the hash floor (branch b). A real change to an in-scope file is silently missed. Fix: when `changed == set()` after `_normalize_dirty_files`, fall through to branch (b) hash floor.

2. **G-3 Gitignore parity** (`test_g6_g3_g10_guards.py`): `_enumerate_live_files` applies only `.tldrignore` patterns via `load_ignore_patterns().match_file()`. It does NOT call `should_ignore()` (which checks git check-ignore). Gitignored files remain in the live set, appear as "new" on every reindex (not in snapshot), and cause false-dirty. Fix: `_enumerate_live_files` must call `should_ignore` the same way `extract_units_from_project` does.

3. **Incr ≡ Full call-graph equivalence** (`test_incr_full_equivalence.py`): Pass-2b in `_build_reapply_call_maps` inserts bare-name keys (e.g., `"save"`) alongside Pass-1 qualified keys (e.g., `"User::save"`) for carried units. The `called_by` invert then has duplicate entries absent from `--full`. ~80% of units diverge, causing 7 re-embeds instead of ≤5. Fix: remove Pass-2b; use `augmented_cache` in `_build_reapply_call_maps`.

4. **S-5 No-rehash on no-op** (`test_s5_no_rehash.py`): `_compute_current_file_hashes` calls `compute_file_hash` for every file on every persist, including no-op runs. Fix: accept `deriver_sha1_map` and `old_snapshot` parameters; use 3-case lookup: case-1 (in deriver_sha1_map), case-2 (in old_snapshot, unchanged), case-3 (fall back to compute_file_hash).

**Guards verified pre-passing:**
- `test_in_scope_hint_still_processes_changed_file`: valid in-scope hint correctly identifies changed file
- `test_subdir_reindex_reuses_unchanged_file`: snapshot keys are scan_path-relative (subdir indexing)
- `test_subdir_noop_reindex_reuses_all_files`: no-op on subdir re-parses nothing
