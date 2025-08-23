"""
Binary Tree Implementation
"""

from collections import deque
from typing import Optional


class TreeNode:
    """Node for binary tree."""

    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"TreeNode({self.val})"


class BinaryTree:
    """Binary tree implementation with common operations."""

    def __init__(self, root: TreeNode | None = None):
        self.root = root

    def inorder_traversal(self) -> list[int]:
        """Inorder traversal: left -> root -> right."""
        result = []

        def inorder(node: TreeNode | None) -> None:
            if node:
                inorder(node.left)
                result.append(node.val)
                inorder(node.right)

        inorder(self.root)
        return result

    def preorder_traversal(self) -> list[int]:
        """Preorder traversal: root -> left -> right."""
        result = []

        def preorder(node: TreeNode | None) -> None:
            if node:
                result.append(node.val)
                preorder(node.left)
                preorder(node.right)

        preorder(self.root)
        return result

    def postorder_traversal(self) -> list[int]:
        """Postorder traversal: left -> right -> root."""
        result = []

        def postorder(node: TreeNode | None) -> None:
            if node:
                postorder(node.left)
                postorder(node.right)
                result.append(node.val)

        postorder(self.root)
        return result

    def level_order_traversal(self) -> list[list[int]]:
        """Level order traversal (BFS)."""
        if not self.root:
            return []

        result = []
        queue = deque([self.root])

        while queue:
            level_size = len(queue)
            level = []

            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(level)

        return result

    def max_depth(self) -> int:
        """Calculate maximum depth of the tree."""

        def depth(node: TreeNode | None) -> int:
            if not node:
                return 0
            return 1 + max(depth(node.left), depth(node.right))

        return depth(self.root)
