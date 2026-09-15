"""
Bug pattern sweep — looks for the SAME CLASS of bug as the
GENE_CLASSES overlap issue, plus a few other common silent-failure
patterns, across the whole codebase.

Patterns checked:
1. Duplicate entries across mutually-exclusive dict-of-lists
   (like GENE_CLASSES) anywhere in the codebase
2. Duplicate keys in dict literals (Python silently keeps the last)
3. Mutable default arguments (classic Python footgun)
4. Bare except clauses that could be silently swallowing errors
5. Hardcoded weight dicts that don't sum to 1.0 (like the
   FIXED_WEIGHTS 0.90 issue mentioned earlier)
6. Functions/variables that shadow builtins
"""

import ast
import sys
from pathlib import Path

ISSUES = []

def flag(filepath, lineno, category, detail):
    ISSUES.append((str(filepath), lineno, category, detail))


def scan_file(filepath):
    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except Exception:
        return

    # --- Pattern 1: dict-of-lists with overlapping values ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            list_values = []
            for k, v in zip(node.keys, node.values):
                if isinstance(v, ast.List):
                    items = []
                    for elt in v.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            items.append(elt.value)
                    if items:
                        key_repr = ast.literal_eval(k) if isinstance(k, ast.Constant) else "?"
                        list_values.append((key_repr, items, v.lineno))

            if len(list_values) >= 2:
                for i in range(len(list_values)):
                    for j in range(i + 1, len(list_values)):
                        cat1, items1, line1 = list_values[i]
                        cat2, items2, line2 = list_values[j]
                        overlap = set(items1) & set(items2)
                        if overlap:
                            flag(filepath, node.lineno,
                                 "DUPLICATE_ACROSS_LISTS",
                                 f"Keys '{cat1}' (line {line1}) and '{cat2}' "
                                 f"(line {line2}) share items: {overlap}")

    # --- Pattern 2: duplicate keys in dict literals ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            seen = {}
            for k in node.keys:
                if isinstance(k, ast.Constant):
                    val = k.value
                    if val in seen:
                        flag(filepath, node.lineno, "DUPLICATE_DICT_KEY",
                             f"Key {val!r} appears more than once "
                             f"(first at line {seen[val]}, again at line {k.lineno})")
                    else:
                        seen[val] = k.lineno

    # --- Pattern 3: mutable default arguments ---
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    flag(filepath, node.lineno, "MUTABLE_DEFAULT_ARG",
                         f"Function '{node.name}' has a mutable default "
                         f"argument ({type(default).__name__})")

    # --- Pattern 4: bare except clauses ---
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            flag(filepath, node.lineno, "BARE_EXCEPT",
                 "Bare 'except:' clause — could silently swallow "
                 "unrelated errors (KeyboardInterrupt, SystemExit, etc.)")

    # --- Pattern 5: weight dicts that should sum to ~1.0 ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Dict):
                keys = []
                vals = []
                all_numeric = True
                for k, v in zip(node.value.keys, node.value.values):
                    if isinstance(k, ast.Constant):
                        keys.append(k.value)
                    if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)) and not isinstance(v.value, bool):
                        vals.append(v.value)
                    else:
                        all_numeric = False
                target_names = [
                    t.id for t in node.targets
                    if isinstance(t, ast.Name)
                ]
                looks_like_weights = any(
                    "weight" in n.lower()
                    for n in target_names
                )
                if looks_like_weights and all_numeric and len(vals) >= 2:
                    total = sum(vals)
                    if abs(total - 1.0) > 0.01:
                        flag(filepath, node.lineno, "WEIGHTS_DONT_SUM_TO_1",
                             f"{target_names} sums to {total:.4f}, not 1.0 "
                             f"(keys={keys}, values={vals})")

    # --- Pattern 6: shadowing builtins ---
    try:
        builtins_set = set(dir(__builtins__))
    except Exception:
        import builtins as _b
        builtins_set = set(dir(_b))

    risky_builtins = builtins_set - {
        "type", "id", "input", "format", "map", "filter", "all", "any",
        "next", "iter", "list", "dict", "set", "str", "int", "float",
        "bool", "len", "min", "max", "sum", "range", "zip", "open",
        "vars", "hash", "object", "property", "super", "round",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in risky_builtins:
                flag(filepath, node.lineno, "SHADOWS_BUILTIN",
                     f"Function named '{node.name}' shadows a Python builtin")


def main():
    root = Path(".")
    py_files = [
        p for p in root.rglob("*.py")
        if "venv" not in p.parts
        and "node_modules" not in p.parts
        and ".git" not in p.parts
        and "site-packages" not in p.parts
        and p.name != "bug_pattern_scan.py"
    ]

    print(f"Scanning {len(py_files)} Python files...\n")

    for f in py_files:
        scan_file(f)

    if not ISSUES:
        print("No issues found by automated patterns.")
        return

    by_category = {}
    for filepath, lineno, category, detail in ISSUES:
        by_category.setdefault(category, []).append((filepath, lineno, detail))

    severity_order = [
        "DUPLICATE_ACROSS_LISTS",
        "DUPLICATE_DICT_KEY",
        "WEIGHTS_DONT_SUM_TO_1",
        "MUTABLE_DEFAULT_ARG",
        "BARE_EXCEPT",
        "SHADOWS_BUILTIN",
    ]

    for cat in severity_order:
        if cat not in by_category:
            continue
        items = by_category[cat]
        print("=" * 70)
        print(f"{cat}  ({len(items)} found)")
        print("=" * 70)
        for filepath, lineno, detail in items:
            print(f"  {filepath}:{lineno}")
            print(f"    {detail}")
        print()

    print(f"\nTOTAL ISSUES FLAGGED: {len(ISSUES)}")
    print("\nNote: some of these may be false positives or intentional —")
    print("review each one, don't blindly 'fix' everything.")


if __name__ == "__main__":
    main()

