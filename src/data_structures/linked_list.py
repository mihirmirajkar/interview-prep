"""
Linked List Implementation
"""

from typing import Optional, Iterator, Any


class ListNode:
    """Node for singly linked list."""
    
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self) -> str:
        return f"ListNode({self.val})"


class LinkedList:
    """Singly linked list implementation."""
    
    def __init__(self):
        self.head: Optional[ListNode] = None
        self.size: int = 0
    
    def append(self, val: int) -> None:
        """Add a new node at the end of the list."""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def prepend(self, val: int) -> None:
        """Add a new node at the beginning of the list."""
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def delete(self, val: int) -> bool:
        """Delete the first occurrence of val. Returns True if deleted."""
        if not self.head:
            return False
        
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        
        if current.next:
            current.next = current.next.next
            self.size -= 1
            return True
        
        return False
    
    def to_list(self) -> list[int]:
        """Convert linked list to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return f"LinkedList({self.to_list()})"
