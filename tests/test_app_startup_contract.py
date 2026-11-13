"""Regression guard for the undefined model_ready startup crash."""
import ast
from pathlib import Path


def test_model_ready_is_assigned_before_first_read():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'app.py').read_text(encoding='utf-8'))
    assigned = set()
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        if isinstance(statement, ast.If) and any(isinstance(node, ast.Name) and node.id == 'model_ready' for node in ast.walk(statement.test)):
            assert 'model_ready' in assigned, 'model_ready must be initialized before page routing'
