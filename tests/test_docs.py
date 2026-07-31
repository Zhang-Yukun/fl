import ast
from pathlib import Path


def test_public_api_has_docstrings():
    package_dir = Path(__file__).parents[1] / "federated_ts"
    missing = []
    for path in package_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            missing.append(f"{path.name}:module")
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not ast.get_docstring(node):
                    missing.append(f"{path.name}:{node.name}")
    assert not missing
