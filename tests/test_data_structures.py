"""Tests for data structures."""

from src.data_structures.linked_list import LinkedList, ListNode
from src.data_structures.binary_tree import BinaryTree, TreeNode


class TestLinkedList:
    """Test cases for linked list implementation."""
    
    def test_append(self):
        """Test appending to linked list."""
        ll = LinkedList()
        ll.append(1)
        ll.append(2)
        ll.append(3)
        
        assert ll.to_list() == [1, 2, 3]
        assert len(ll) == 3
    
    def test_prepend(self):
        """Test prepending to linked list."""
        ll = LinkedList()
        ll.prepend(1)
        ll.prepend(2)
        ll.prepend(3)
        
        assert ll.to_list() == [3, 2, 1]
        assert len(ll) == 3
    
    def test_delete(self):
        """Test deleting from linked list."""
        ll = LinkedList()
        ll.append(1)
        ll.append(2)
        ll.append(3)
        
        assert ll.delete(2) is True
        assert ll.to_list() == [1, 3]
        assert len(ll) == 2
        
        assert ll.delete(5) is False
        assert ll.to_list() == [1, 3]
    
    def test_empty_list(self):
        """Test operations on empty list."""
        ll = LinkedList()
        assert ll.to_list() == []
        assert len(ll) == 0
        assert ll.delete(1) is False


class TestBinaryTree:
    """Test cases for binary tree implementation."""
    
    def setup_method(self):
        """Set up test tree."""
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        self.root = TreeNode(1)
        self.root.left = TreeNode(2)
        self.root.right = TreeNode(3)
        self.root.left.left = TreeNode(4)
        self.root.left.right = TreeNode(5)
        self.tree = BinaryTree(self.root)
    
    def test_inorder_traversal(self):
        """Test inorder traversal."""
        result = self.tree.inorder_traversal()
        assert result == [4, 2, 5, 1, 3]
    
    def test_preorder_traversal(self):
        """Test preorder traversal."""
        result = self.tree.preorder_traversal()
        assert result == [1, 2, 4, 5, 3]
    
    def test_postorder_traversal(self):
        """Test postorder traversal."""
        result = self.tree.postorder_traversal()
        assert result == [4, 5, 2, 3, 1]
    
    def test_level_order_traversal(self):
        """Test level order traversal."""
        result = self.tree.level_order_traversal()
        assert result == [[1], [2, 3], [4, 5]]
    
    def test_max_depth(self):
        """Test maximum depth calculation."""
        assert self.tree.max_depth() == 3
    
    def test_empty_tree(self):
        """Test operations on empty tree."""
        empty_tree = BinaryTree()
        assert empty_tree.inorder_traversal() == []
        assert empty_tree.level_order_traversal() == []
        assert empty_tree.max_depth() == 0
