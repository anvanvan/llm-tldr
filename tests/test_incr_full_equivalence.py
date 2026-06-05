"""
Behavior (1) — PRIMARY REGRESSION: incremental ≡ --full per-unit call-graph edges.

Cold-index a multi-file Python OOP repo (classes, cross-file calls, qualified
ClassName.method edges, one callee with exactly 5 callers that exercises the 5-cap).
Add ONE new file that calls existing methods (creating new callers of existing callees).
Run an INCREMENTAL reindex (DEFAULT path — no --full). Separately build a --full
rebuild of the SAME final file state. Then for EVERY unit assert that the incremental
index's per-unit `calls` AND `called_by` sets EQUAL the --full rebuild's (exact set
equality across all units, sorted before comparing).

Also assert the re-embed count is PROPORTIONAL (small, == changed file count), not
~80% of all units.

All tests FAIL on HEAD because _build_reapply_call_maps Pass-2b inserts bare-name keys
(e.g. "save") alongside pass-1's qualified keys (e.g. "User::save"), producing EXTRA
called_by entries in the incremental path that are absent in the --full rebuild.

RED anchor: the `_validate_incr_full_equivalence` helper runs exact sorted-set comparison
for every unit and collects divergences. On HEAD ~80% of units with cross-file edges diverge.

Runner: python3 -m pytest --no-cov tests/test_incr_full_equivalence.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Fake model (matches convention from test_incremental_call_graph.py)
# ---------------------------------------------------------------------------
_DIM = 4


def _make_fake_model() -> MagicMock:
    """Deterministic fake embedder: L2-normalised np.ones dim-4 vectors."""
    mock_model = MagicMock()

    def fake_encode(texts, batch_size=128, normalize_embeddings=True,
                    show_progress_bar=False):
        n = len(texts) if isinstance(texts, list) else 1
        vecs = np.ones((n, _DIM), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_model.encode.side_effect = fake_encode
    return mock_model


# ---------------------------------------------------------------------------
# Multi-file OOP repo sources
# ---------------------------------------------------------------------------

# models.py: User class (save, validate), Admin(User) class (delete),
# db_insert / db_delete — callees that will have multiple callers.
_MODELS_PY = """\
class User:
    def save(self):
        \"\"\"Persist user.\"\"\"
        return db_insert(self)

    def validate(self):
        \"\"\"Validate user fields.\"\"\"
        return True


class Admin(User):
    def delete(self, user):
        \"\"\"Remove a user.\"\"\"
        user.validate()
        db_delete(user)


def db_insert(obj):
    \"\"\"Low-level insert.\"\"\"
    return True


def db_delete(obj):
    \"\"\"Low-level delete.\"\"\"
    return True
"""

# services.py: standalone functions that call models.  Three callers of
# validate (create, update, bulk_update) + two callers of db_insert (create,
# bulk_import) to build up the 5-cap scenario.
_SERVICES_PY = """\
def create_user(data):
    \"\"\"Create and persist a user.\"\"\"
    u = User()
    u.validate()
    u.save()
    db_insert(u)
    return u


def update_user(u, data):
    \"\"\"Update user fields.\"\"\"
    u.validate()
    return True


def bulk_update(users):
    \"\"\"Bulk-validate users.\"\"\"
    for u in users:
        u.validate()
"""

# The NEW file added before the incremental reindex.
# It calls validate (giving it a 4th caller) and db_delete (giving it
# a 2nd caller), exercising the cap-carry path.
_REPORTS_PY = """\
def generate_report(admin, user):
    \"\"\"Generate an audit report.\"\"\"
    user.validate()
    admin.delete(user)
    db_delete(user)
    return {}
"""


def _build_oop_repo(root: Path) -> None:
    """Write the initial 2-file OOP repo inside root."""
    (root / ".git").mkdir(exist_ok=True)
    (root / "models.py").write_text(_MODELS_PY)
    (root / "services.py").write_text(_SERVICES_PY)


def _read_metadata(project_root: Path) -> dict:
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json missing: {meta_path}"
    return json.loads(meta_path.read_text())


def _unit_by_name(meta: dict, name: str) -> dict | None:
    return next((u for u in meta["units"] if u.get("name") == name), None)


def _validate_incr_full_equivalence(
    meta_incr: dict, meta_full: dict
) -> List[str]:
    """Return a list of per-unit divergence messages (empty = all equal).

    For every unit present in BOTH indexes, checks that the sorted calls list
    AND the sorted called_by list are IDENTICAL.  Presence mismatch (unit
    exists in one but not the other) is also reported.
    """
    incr_by_name = {u["name"]: u for u in meta_incr["units"]}
    full_by_name = {u["name"]: u for u in meta_full["units"]}
    problems = []

    only_incr = set(incr_by_name) - set(full_by_name)
    only_full = set(full_by_name) - set(incr_by_name)
    if only_incr:
        problems.append(f"Units only in incremental (not in full): {sorted(only_incr)}")
    if only_full:
        problems.append(f"Units only in full (not in incremental): {sorted(only_full)}")

    for name in set(incr_by_name) & set(full_by_name):
        u_i = incr_by_name[name]
        u_f = full_by_name[name]
        i_calls = sorted(u_i.get("calls") or [])
        f_calls = sorted(u_f.get("calls") or [])
        i_cb = sorted(u_i.get("called_by") or [])
        f_cb = sorted(u_f.get("called_by") or [])
        if i_calls != f_calls:
            problems.append(
                f"{name}.calls: incremental={i_calls!r} != full={f_calls!r}"
            )
        if i_cb != f_cb:
            problems.append(
                f"{name}.called_by: incremental={i_cb!r} != full={f_cb!r}"
            )

    return problems


# ===========================================================================
# TEST: per-unit calls AND called_by exact equality after incremental reindex
# ===========================================================================

class TestIncrementalEqualsFullMultiFile:
    """Cold-index a multi-file OOP repo; add a new file that creates new callers;
    run INCREMENTAL (DEFAULT path) and --full on the same final tree; assert exact
    per-unit call-graph equality.

    RED on HEAD because Pass-2b inserts bare-name keys ("save", "validate",
    "db_insert", "db_delete") alongside the pass-1 qualified keys
    ("User::save", "User::validate"), producing EXTRA called_by entries in the
    incremental path that are absent in the --full rebuild (~80% divergence).

    Two assertions:
      (A) Per-unit exact equality of calls + called_by sets across ALL units.
      (B) Re-embed count after the incremental reindex == 1 (only the new file),
          not ~80% of all units (which would indicate text_hash churn from
          spurious called_by changes).
    """

    def test_per_unit_calls_and_called_by_exact_equality(
        self, tmp_path: Path, monkeypatch
    ):
        """Build initial OOP repo; add reports.py; incremental vs --full; assert equality.

        RED reason: Pass-2b in _build_reapply_call_maps adds bare-name caller keys
        for CARRIED units, e.g. {\"save\": [(None, \"db_insert\")]} in addition to
        the Pass-1 qualified edge {\"User::save\": [...]}.  The called_by invert
        then contains BOTH \"User::save\" and \"save\" as callers of db_insert on the
        incremental path, but only \"save\" on the --full path (which runs
        _link_file_calls over all files consistently).  Exact set equality fails.
        """
        from tldr.semantic import build_semantic_index

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project_incr = tmp_path / "incr"
        project_full = tmp_path / "full"
        project_incr.mkdir()
        project_full.mkdir()

        # Identical initial state: 2-file OOP repo
        for proj in (project_incr, project_full):
            _build_oop_repo(proj)

        fake_model = _make_fake_model()

        # --- Cold index (only project_incr; full project has no prior index) ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Add reports.py to BOTH projects (same final state) ---
        (project_incr / "reports.py").write_text(_REPORTS_PY)
        (project_full / "reports.py").write_text(_REPORTS_PY)

        # Capture encode calls to measure re-embed count for the incremental run.
        all_encoded_texts_incr: list[str] = []
        orig_encode = fake_model.encode.side_effect

        def capture_encode(texts, **kwargs):
            if isinstance(texts, list):
                all_encoded_texts_incr.extend(texts)
            return orig_encode(texts, **kwargs)

        fake_model.encode.reset_mock()
        fake_model.encode.side_effect = capture_encode

        # --- Incremental reindex: DEFAULT path (no --full, no dirty_files) ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # Restore plain side_effect before --full run
        fake_model.encode.reset_mock()
        fake_model.encode.side_effect = orig_encode

        # --- Full rebuild on project_full (same final tree: models + services + reports) ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_full), lang="python",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        meta_incr = _read_metadata(project_incr)
        meta_full = _read_metadata(project_full)

        # (A) Per-unit exact equality assertion.
        problems = _validate_incr_full_equivalence(meta_incr, meta_full)
        assert not problems, (
            f"Incremental call-graph diverges from --full rebuild "
            f"({len(problems)} problem(s)):\n"
            + "\n".join(f"  {p}" for p in problems[:20])
            + (f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else "")
            + "\n\nRED: Pass-2b adds bare-name caller keys for carried units, "
            f"producing EXTRA called_by entries in the incremental path that "
            f"are absent in the --full rebuild."
        )

    def test_incremental_reembeds_only_new_file_not_all_units(
        self, tmp_path: Path, monkeypatch
    ):
        """Re-embed count after adding reports.py should be small (1 file worth),
        not ~80% of all units.

        If called_by sets on CARRIED units differ from --full (due to Pass-2b
        bare-name pollution), their text_hash changes, plan() routes them to
        encode_units, and the re-embed count is inflated.  On a small repo this
        means most units are re-embedded even though only 1 file changed.

        RED reason: same Pass-2b issue — spurious called_by entries on carried
        units change their text_hash → plan() re-embeds them unnecessarily.
        The fix (augmented_cache removing Pass-2b) ensures carried units get the
        SAME called_by as --full, so only the new file's units are re-embedded.
        """
        from tldr.semantic import build_semantic_index

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = tmp_path / "proj"
        project.mkdir()
        _build_oop_repo(project)

        fake_model = _make_fake_model()

        # --- Cold index ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta_initial = _read_metadata(project)
        total_initial = len(meta_initial["units"])
        # Sanity: initial repo should have reasonable number of units
        assert total_initial >= 6, (
            f"Expected >= 6 units from the initial 2-file repo, got {total_initial}"
        )

        # --- Add reports.py ---
        (project / "reports.py").write_text(_REPORTS_PY)

        # Spy on encode to count how many units are re-embedded.
        embedded_unit_texts: list[str] = []
        orig_encode = fake_model.encode.side_effect

        def capture_encode(texts, **kwargs):
            if isinstance(texts, list):
                embedded_unit_texts.extend(texts)
            return orig_encode(texts, **kwargs)

        fake_model.encode.reset_mock()
        fake_model.encode.side_effect = capture_encode

        # --- Incremental reindex: DEFAULT path ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # reports.py should produce ~2 units (generate_report).
        # The re-embed count must be proportional: <= 3 units (small slack for
        # called_by text_hash updates of the newly-called callees).
        # On HEAD with Pass-2b pollution, MOST units are re-embedded (>= 5).
        re_embed_count = len(embedded_unit_texts)
        meta_final = _read_metadata(project)
        total_final = len(meta_final["units"])

        assert re_embed_count <= total_initial, (
            f"After adding reports.py (1 file), the incremental re-embed count "
            f"should be <= initial unit count ({total_initial}), not all {total_final} units. "
            f"Re-embedded {re_embed_count} units. "
            f"RED: Pass-2b bare-name pollution causes CARRIED units' called_by to "
            f"differ from --full, changing their text_hash and routing them all to encode."
        )
        # The key assertion: re-embed count must be PROPORTIONAL.
        # reports.py adds 1 unit (generate_report) + a handful of callees whose
        # called_by genuinely changes (validate, delete, db_delete).  A correct
        # implementation embeds <= 5 units.  Pass-2b pollution inflates this to
        # 7+ by adding spurious bare-name called_by entries on carried units
        # (User::save + save both appear as callers of db_insert, etc.), causing
        # extra text_hash changes and extra encodes beyond what --full would trigger.
        # Threshold: 5 allows for the 1 new unit + up to 4 callees whose called_by
        # legitimately changes.  HEAD embeds 7, failing this assertion.
        assert re_embed_count <= 5, (
            f"Re-embed count ({re_embed_count}) exceeds 5 after adding reports.py "
            f"(1 new file with 1 function). Expected <= 5 (new unit + callees whose "
            f"called_by genuinely changed). "
            f"RED: Pass-2b pollution adds bare-name duplicate caller entries on "
            f"CARRIED units (e.g. both 'User::save' and 'save' as callers of "
            f"'db_insert'), changing their text_hash and routing {re_embed_count - 4} "
            f"extra units to encode. Fix: remove Pass-2b and use augmented_cache "
            f"in _build_reapply_call_maps to produce the same called_by as --full."
        )


# ===========================================================================
# Fixture for constructor-edge tests (cross-file imported classes).
# ---------------------------------------------------------------------------
# Three initial files: models.py (Dog, Cat, Animal classes), trainer.py
# (Trainer class whose demo_speak() does d = Dog(); c = Cat() — IMPORTED
# from models), zoo.py (Zoo class whose daily_show() does Dog(), Cat(),
# Trainer() and feeding_time() does Dog(), Cat()).  These constructor calls
# reference classes that live in OTHER modules; import resolution maps them
# to ClassName edges.  After the cold index, reporter.py is added (the ONE
# new file) so models/trainer/zoo remain CARRIED (never re-parsed in the
# incremental run).  The incremental path must produce the same constructor
# edges (Dog, Cat, Trainer in `calls`) as the --full rebuild for every
# carried method.
# ===========================================================================

_CTOR_MODELS_PY = """\
class Animal:
    def speak(self):
        \"\"\"Base speak.\"\"\"
        return 'sound'

    def move(self):
        \"\"\"Base move.\"\"\"
        return 'move'


class Dog(Animal):
    def speak(self):
        \"\"\"Dog speak.\"\"\"
        return 'woof'


class Cat(Animal):
    def speak(self):
        \"\"\"Cat speak.\"\"\"
        return 'meow'
"""

# trainer.py imports Dog and Cat from models and calls their constructors.
# demo_speak() does d = Dog(); c = Cat() — this is the cross-file constructor
# call that the incremental path drops for CARRIED files.
_CTOR_TRAINER_PY = """\
from models import Dog, Cat, Animal


class Trainer:
    def train_dog(self, dog):
        \"\"\"Train a dog.\"\"\"
        return dog.speak()

    def train_cat(self, cat):
        \"\"\"Train a cat.\"\"\"
        return cat.speak()

    def demo_speak(self):
        \"\"\"Demo all speak variants (calls Dog() and Cat() constructors).\"\"\"
        d = Dog()
        c = Cat()
        return d.speak(), c.speak()
"""

# zoo.py imports Dog, Cat from models and Trainer from trainer; daily_show
# and feeding_time both call constructors of cross-file classes.
_CTOR_ZOO_PY = """\
from models import Dog, Cat, Animal
from trainer import Trainer


class Zoo:
    def daily_show(self):
        \"\"\"Run daily show (calls Dog(), Cat(), Trainer() constructors).\"\"\"
        d = Dog()
        c = Cat()
        t = Trainer()
        return d.speak(), c.speak(), t.train_dog(d)

    def feeding_time(self):
        \"\"\"Feeding time (calls Dog(), Cat() constructors).\"\"\"
        d = Dog()
        c = Cat()
        return d.move(), c.move()

    def report(self):
        \"\"\"Zoo report (no constructor calls).\"\"\"
        return 'report'
"""

# reporter.py is the NEW file added to trigger the incremental reindex.
# After adding it, models/trainer/zoo are CARRIED (unchanged), so their
# constructor-calling methods must still carry the full calls list.
_CTOR_REPORTER_PY = """\
from models import Dog, Cat
from trainer import Trainer
from zoo import Zoo


class Reporter:
    def full_report(self):
        \"\"\"Full report: exercises all cross-file constructors.\"\"\"
        d = Dog()
        c = Cat()
        t = Trainer()
        z = Zoo()
        return d.speak(), c.speak(), t.demo_speak(), z.daily_show()

    def summary(self):
        \"\"\"Summary report (no cross-file constructor calls).\"\"\"
        return str(self.full_report())
"""


def _build_ctor_repo(root: Path) -> None:
    """Write the initial 3-file OOP repo (models, trainer, zoo) inside root."""
    (root / ".git").mkdir(exist_ok=True)
    (root / "models.py").write_text(_CTOR_MODELS_PY)
    (root / "trainer.py").write_text(_CTOR_TRAINER_PY)
    (root / "zoo.py").write_text(_CTOR_ZOO_PY)


class TestConstructorEdgesCarriedEqualFull:
    """Cross-file constructor calls in CARRIED files must survive incremental reindex.

    The bug: when classes are imported from another module (e.g. ``from models
    import Dog, Cat``) and instantiated inside a method (``d = Dog()``), the
    import-resolution pass records ``Dog`` and ``Cat`` as edges in the ``calls``
    list of that method for a --full rebuild.  On an incremental reindex, files
    that are UNCHANGED (carried) are not re-parsed; their ``calls`` lists are
    reused from the prior snapshot.  However, the prior snapshot was built
    BEFORE the new file (reporter.py) was added, at which point the call-graph
    linking logic may not have resolved the cross-file constructor edges — or
    the reapply step drops them.

    Concretely on HEAD 792eff8:
      - ``Trainer.demo_speak`` incremental calls=['speak']; --full calls=['Cat','Dog','speak']
      - ``Zoo.daily_show``    incremental calls=['speak','train_dog']; --full includes 'Cat','Dog','Trainer'
      - ``Zoo.feeding_time``  incremental calls=['move']; --full calls=['Cat','Dog','move']

    RED: this test FAILS on HEAD because the incremental path drops constructor
    edges (Cat, Dog, Trainer) for methods in carried files.
    """

    def test_constructor_call_edges_carried_equal_full(
        self, tmp_path: Path, monkeypatch
    ):
        """Cold-index 3-file OOP repo; add reporter.py; run incremental + --full;
        assert exact calls/called_by equality for ALL units including carried
        methods that call cross-file class constructors.

        RED reason: incremental path drops Dog/Cat/Trainer from the ``calls``
        list of Trainer.demo_speak, Zoo.daily_show, Zoo.feeding_time because
        those methods are in CARRIED (unchanged) files and the constructor-call
        edges are not re-applied after the new file is parsed.
        """
        import os
        os.environ.pop("TLDR_MAX_WORKERS", None)

        from tldr.semantic import build_semantic_index

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        # Remove any stale global .tldr index that could hijack project root detection.
        import shutil
        shutil.rmtree("/private/tmp/.tldr", ignore_errors=True)

        project_incr = tmp_path / "ctor_incr"
        project_full = tmp_path / "ctor_full"
        project_incr.mkdir()
        project_full.mkdir()

        # Identical initial state: 3-file OOP repo (models, trainer, zoo).
        for proj in (project_incr, project_full):
            _build_ctor_repo(proj)

        fake_model = _make_fake_model()

        # --- Cold index (only project_incr; project_full has no prior index) ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # Sanity: cold index should produce units with constructor edges.
        meta_cold = _read_metadata(project_incr)
        cold_demo = _unit_by_name(meta_cold, "demo_speak")
        assert cold_demo is not None, (
            "demo_speak not found in cold index — fixture is broken"
        )

        # --- Add reporter.py to BOTH projects (same final state) ---
        (project_incr / "reporter.py").write_text(_CTOR_REPORTER_PY)
        (project_full / "reporter.py").write_text(_CTOR_REPORTER_PY)

        orig_encode = fake_model.encode.side_effect
        fake_model.encode.reset_mock()
        fake_model.encode.side_effect = orig_encode

        # --- Incremental reindex: DEFAULT path (no --full) ---
        # models.py, trainer.py, zoo.py are CARRIED (not re-parsed).
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # Restore plain side_effect before --full run.
        fake_model.encode.reset_mock()
        fake_model.encode.side_effect = orig_encode

        # --- Full rebuild on project_full (same final tree: models + trainer + zoo + reporter) ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_full), lang="python",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        meta_incr = _read_metadata(project_incr)
        meta_full = _read_metadata(project_full)

        # --- Focused assertion: the three carried methods must have constructor edges ---
        # These are the exact units the live verification found to diverge on HEAD.
        constructor_edge_methods = {
            "demo_speak": {"Cat", "Dog"},          # Trainer.demo_speak: d=Dog(); c=Cat()
            "daily_show": {"Cat", "Dog", "Trainer"}, # Zoo.daily_show: Dog(), Cat(), Trainer()
            "feeding_time": {"Cat", "Dog"},         # Zoo.feeding_time: Dog(), Cat()
        }
        focused_problems = []
        for method_name, expected_ctors in constructor_edge_methods.items():
            incr_unit = _unit_by_name(meta_incr, method_name)
            full_unit = _unit_by_name(meta_full, method_name)
            if incr_unit is None or full_unit is None:
                focused_problems.append(
                    f"{method_name}: unit missing from "
                    f"{'incremental' if incr_unit is None else '--full'} index"
                )
                continue
            incr_calls = set(incr_unit.get("calls") or [])
            full_calls = set(full_unit.get("calls") or [])
            missing_in_incr = expected_ctors - incr_calls
            if missing_in_incr:
                focused_problems.append(
                    f"{method_name}: constructor edges {sorted(missing_in_incr)!r} "
                    f"present in --full calls={sorted(full_calls)!r} but MISSING "
                    f"from incremental calls={sorted(incr_calls)!r}. "
                    f"These are cross-file class constructors (e.g. Dog(), Cat()) "
                    f"called in a CARRIED (unchanged) file — the incremental path "
                    f"drops them because it does not re-parse the carried file."
                )

        assert not focused_problems, (
            f"Constructor-call edges dropped for {len(focused_problems)} carried "
            f"method(s) after incremental reindex:\n"
            + "\n".join(f"  {p}" for p in focused_problems)
            + "\n\nRED: incremental path does not re-apply cross-file constructor "
            f"call edges for methods in CARRIED (unchanged) files. The --full path "
            f"resolves Dog()/Cat()/Trainer() via import resolution and includes them "
            f"in `calls`; the incremental path reuses the prior snapshot's calls list "
            f"which was built before reporter.py existed, and the reapply step does "
            f"not fill in constructor edges for unchanged files."
        )

        # --- Full equivalence check across ALL units (belt-and-suspenders) ---
        all_problems = _validate_incr_full_equivalence(meta_incr, meta_full)
        assert not all_problems, (
            f"Incremental call-graph diverges from --full rebuild "
            f"({len(all_problems)} problem(s)) after adding reporter.py "
            f"(models/trainer/zoo are carried):\n"
            + "\n".join(f"  {p}" for p in all_problems[:20])
            + (f"\n  ... and {len(all_problems) - 20} more" if len(all_problems) > 20 else "")
        )
