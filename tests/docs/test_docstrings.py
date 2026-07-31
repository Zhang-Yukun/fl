import ast
from pathlib import Path


def _check_python_tree(root: Path):
    missing = []
    for path in root.rglob("*.py"):
        if "reference_patchtst" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            missing.append(f"{path.relative_to(root)}:module")
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not ast.get_docstring(node):
                    missing.append(f"{path.relative_to(root)}:{node.name}")
    return missing


def test_public_api_has_docstrings():
    src_dir = Path(__file__).parents[2]
    missing = _check_python_tree(src_dir / "federated_ts")
    missing.extend(f"scripts/{item}" for item in _check_python_tree(src_dir / "scripts"))
    assert not missing
