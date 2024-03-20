from collections.abc import Callable
from typing import Any


def tree_flatten(tree: Any, prefix: str = "", is_leaf: Callable = None) -> list[tuple[str, Any]]:
    flat_tree = []

    if is_leaf is None or not is_leaf(tree):
        if isinstance(tree, list | tuple):
            for i, t in enumerate(tree):
                flat_tree.extend(tree_flatten(t, prefix=f"{prefix}.{i}", is_leaf=is_leaf))
            return flat_tree
        if isinstance(tree, dict):
            for k, t in tree.items():
                flat_tree.extend(tree_flatten(t, prefix=f"{prefix}.{k}", is_leaf=is_leaf))
            return flat_tree
    return [(prefix[1:], tree)]
