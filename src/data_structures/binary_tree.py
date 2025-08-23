"""
Binary Tree Implementation
"""

from typing import Optional, List, Iterator
from collections import deque


class TreeNode:
    """Node for binary tree."""
    
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self) -> str:
        return f"TreeNode({self.val})"


class BinaryTree:
    """Binary tree implementation with common operations."""
    
    def __init__(self, root: Optional[TreeNode] = None):
        self.root = root
    
    def inorder_traversal(self) -> List[int]:
        """Inorder traversal: left -> root -> right."""
        result = []
        
        def inorder(node: Optional[TreeNode]) -> None:
            if node:
                inorder(node.left)
                result.append(node.val)
                inorder(node.right)
        
        inorder(self.root)
        return result
    
    def preorder_traversal(self) -> List[int]:
        """Preorder traversal: root -> left -> right."""
        result = []
        
        def preorder(node: Optional[TreeNode]) -> None:
            if node:
                result.append(node.val)
                preorder(node.left)
                preorder(node.right)
        
        preorder(self.root)
        return result
    
    def postorder_traversal(self) -> List[int]:
        """Postorder traversal: left -> right -> root."""
        result = []
        
        def postorder(node: Optional[TreeNode]) -> None:
            if node:
                postorder(node.left)
                postorder(node.right)
                result.append(node.val)
        
        postorder(self.root)
        return result
    
    def level_order_traversal(self) -> List[List[int]]:
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
        def depth(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            return 1 + max(depth(node.left), depth(node.right))
        
        return depth(self.root)
