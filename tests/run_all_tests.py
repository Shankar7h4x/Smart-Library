"""Master test runner to execute all test functions across the tests directory."""

from datetime import datetime
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TESTS_DIR = ROOT / "tests"


class MockMonkeyPatch:
    def __init__(self):
        self._env_undo = {}
        self._attr_undo = []

    def setenv(self, name, value):
        if name not in self._env_undo:
            self._env_undo[name] = os.environ.get(name)
        os.environ[name] = value

    def delenv(self, name, raising=False):
        if name not in self._env_undo:
            self._env_undo[name] = os.environ.get(name)
        if name in os.environ:
            del os.environ[name]

    def setattr(self, target, name, value, raising=True):
        if isinstance(target, str):
            mod_name, attr_name = target.rsplit(".", 1)
            target = sys.modules[mod_name]
            name = attr_name
        self._attr_undo.append((target, name, hasattr(target, name), getattr(target, name, None)))
        setattr(target, name, value)

    def undo(self):
        for k, v in self._env_undo.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for target, name, had_attr, old_val in reversed(self._attr_undo):
            if had_attr:
                setattr(target, name, old_val)
            elif hasattr(target, name):
                delattr(target, name)


def run_all_tests():
    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    total_passed = 0
    total_failed = 0
    failed_tests = []

    print("=" * 70)
    print("RUNNING ALL AUTOMATED TESTS")
    print("=" * 70)

    for test_path in test_files:
        module_name = test_path.stem
        spec = importlib.util.spec_from_file_location(module_name, str(test_path))
        if spec is None or spec.loader is None:
            print(f"Skipping {test_path.name}: Cannot create spec")
            continue

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"\n❌ FAILED TO IMPORT {test_path.name}: {e}")
            traceback.print_exc()
            total_failed += 1
            failed_tests.append((f"{test_path.name}:__import__", str(e)))
            continue

        test_funcs = [
            (name, fn)
            for name, fn in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_") and fn.__module__ == module_name
        ]

        if not test_funcs:
            continue

        print(f"\n📁 {test_path.name} ({len(test_funcs)} tests):")
        for name, fn in test_funcs:
            sig = inspect.signature(fn)
            kwargs = {}
            temp_dir = None
            mp = None

            if "monkeypatch" in sig.parameters:
                mp = MockMonkeyPatch()
                kwargs["monkeypatch"] = mp
            if "tmp_path" in sig.parameters:
                temp_dir = tempfile.mkdtemp(prefix="test_tmp_")
                kwargs["tmp_path"] = Path(temp_dir)

            try:
                fn(**kwargs)
                print(f"   ✓ {name}")
                total_passed += 1
            except Exception as e:
                print(f"   ❌ {name}: {e}")
                traceback.print_exc()
                total_failed += 1
                failed_tests.append((f"{test_path.name}:{name}", str(e)))
            finally:
                if mp:
                    mp.undo()
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass

    print("\n" + "=" * 70)
    print(f"SUMMARY: {total_passed} passed, {total_failed} failed (Total: {total_passed + total_failed})")
    print("=" * 70)

    if failed_tests:
        print("\nFailed Tests:")
        for tname, err in failed_tests:
            print(f"  - {tname}: {err}")
        return False
    else:
        print("ALL TESTS PASSED WITH ZERO REGRESSIONS! 🚀")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
