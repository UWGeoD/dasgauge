"""Verify dasgauge is fully independent of the DAS_Preprocessing source project.

These tests guard the provenance guarantee: dasgauge contains *derived* code,
never a runtime dependency on Project A.
"""

import ast
import pathlib
import subprocess
import sys
import unittest

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
PY_FILES = sorted((PKG_ROOT / "dasgauge").rglob("*.py"))

# Top-level module names that exist only inside Project A's repository root.
# Importing any of them would make dasgauge depend on Project A at runtime.
PROJECT_A_MODULES = {
    "DAS_Preprocessing",
    "Utilities",
    "DAS",
    "dataset",
    "experiment_config",
    "prepare_data",
    "sweep_paths",
    "config",
}

# Substrings that would indicate a local working copy or sys.path tampering.
# The repository *name* is deliberately NOT listed: it legitimately appears in
# the attribution docstrings that PROVENANCE.md requires.
FORBIDDEN_SUBSTRINGS = [
    "/home/",      # absolute Linux home path
    "/Users/",     # absolute macOS home path
    "/tmp/",       # temporary extraction clone
    "sys.path.insert",
    "sys.path.append",
]


def _imported_module_roots(tree):
    """Top-level names of every module imported by an AST, absolute imports only."""
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # node.level > 0 is a relative (intra-package) import — always fine
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


class SourceHygieneTests(unittest.TestCase):
    def test_package_has_python_files(self):
        self.assertTrue(PY_FILES, "no dasgauge modules found")

    def test_no_imports_from_project_a(self):
        for path in PY_FILES:
            tree = ast.parse(path.read_text())
            leaked = _imported_module_roots(tree) & PROJECT_A_MODULES
            with self.subTest(file=path.name):
                self.assertEqual(leaked, set(), f"{path.name} imports {leaked}")

    @unittest.skipUnless(
        hasattr(sys, "stdlib_module_names"), "sys.stdlib_module_names needs Python 3.10+"
    )
    def test_only_declared_third_party_imports(self):
        """Third-party imports are limited to the declared dependencies."""
        allowed = {"numpy", "scipy", "h5py", "matplotlib"}
        stdlib = set(sys.stdlib_module_names)
        for path in PY_FILES:
            tree = ast.parse(path.read_text())
            extra = _imported_module_roots(tree) - allowed - stdlib - {"dasgauge"}
            with self.subTest(file=path.name):
                self.assertEqual(extra, set(), f"{path.name} imports undeclared {extra}")

    def test_no_local_paths_or_syspath_tampering(self):
        for path in PY_FILES:
            text = path.read_text()
            for needle in FORBIDDEN_SUBSTRINGS:
                with self.subTest(file=path.name, needle=needle):
                    self.assertNotIn(needle, text)

    def test_no_hard_coded_path_constants(self):
        for path in PY_FILES:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                with self.subTest(file=path.name):
                    self.assertFalse(
                        stripped.startswith(("PATH =", "ROOT =", "DATA_DIR =")),
                        f"{path.name}: hard-coded path constant: {stripped}",
                    )

    def test_attribution_present_in_derived_modules(self):
        """Every module with derived code names the exact source commit."""
        sha = "896e00005b68c7878ae80d50833bd9eeabe7ebc7"
        for name in ("io.py", "preprocessing.py", "plotting.py"):
            doc = ast.get_docstring(ast.parse((PKG_ROOT / "dasgauge" / name).read_text()))
            with self.subTest(file=name):
                self.assertIsNotNone(doc)
                self.assertIn("DAS_Preprocessing", doc)
                self.assertIn(sha, doc)
                self.assertIn("PROVENANCE.md", doc)


class RuntimeIndependenceTests(unittest.TestCase):
    def test_imports_in_clean_subprocess(self):
        """dasgauge imports with only its own repo on sys.path (no Project A)."""
        code = (
            "import dasgauge, sys;"
            "mods = [m for m in sys.modules if 'DAS_Preprocessing' in m or m in "
            "('Utilities', 'DAS')];"
            "assert not mods, mods;"
            "assert dasgauge.DAS and dasgauge.make_preprocess and dasgauge.plot_das_data;"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PKG_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_dependency_declaration_excludes_project_a(self):
        pyproject = (PKG_ROOT / "pyproject.toml").read_text()
        self.assertNotIn("DAS_Preprocessing", pyproject)
        for pkg in ("numpy", "scipy", "h5py"):
            self.assertIn(pkg, pyproject)

    def test_no_submodule(self):
        self.assertFalse(
            (PKG_ROOT / ".gitmodules").exists(), "Project A must not be a submodule"
        )

    def test_provenance_records_exist(self):
        self.assertTrue((PKG_ROOT / "PROVENANCE.md").is_file())
        self.assertTrue((PKG_ROOT / "provenance" / "DAS_Preprocessing.json").is_file())


if __name__ == "__main__":
    unittest.main()
