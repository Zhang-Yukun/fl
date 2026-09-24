import ast
from pathlib import Path


def _check_python_tree(root: Path):
    missing = []
    for path in root.rglob("*.py"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        if "reference_patchtst" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            missing.append(f"{path.relative_to(root)}:module")
        exported_names = set()
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                continue
            try:
                exported_names.update(ast.literal_eval(node.value))
            except (TypeError, ValueError):
                pass
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in exported_names and not ast.get_docstring(node):
                    missing.append(f"{path.relative_to(root)}:{node.name}")
    return missing


def test_public_api_has_docstrings():
    src_dir = Path(__file__).parents[2]
    missing = _check_python_tree(src_dir / "fedlab")
    missing.extend(f"scripts/{item}" for item in _check_python_tree(src_dir / "scripts"))
    assert not missing
