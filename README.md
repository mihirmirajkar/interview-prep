# Interview Preparation in Python

A comprehensive Python project for practicing coding interview questions with modern tooling and best practices.

## 🚀 Features

- **Modern Python Setup**: Python 3.12+ with latest tooling
- **Code Quality**: Pre-commit hooks, black formatting, ruff linting
- **Testing**: pytest with coverage reporting
- **Type Safety**: Full type hint support with mypy
- **Organization**: Problems categorized by data structures and algorithms
- **Documentation**: Comprehensive docstrings and README files

## 📁 Project Structure

```
interview-prep/
├── src/
│   ├── __init__.py
│   ├── data_structures/     # Custom data structure implementations
│   ├── algorithms/          # Algorithm implementations
│   └── utils/              # Helper utilities
├── problems/
│   ├── arrays/             # Array-based problems
│   ├── strings/            # String manipulation problems
│   ├── linked_lists/       # Linked list problems
│   ├── trees/              # Tree and graph problems
│   ├── dynamic_programming/ # DP problems
│   └── sorting_searching/   # Sorting and searching problems
├── tests/                  # Test files
├── docs/                   # Documentation
└── scripts/               # Utility scripts
```

## 🛠️ Setup

1. **Clone the repository** (after pushing to GitHub):
   ```bash
   git clone <your-repo-url>
   cd interview-prep
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_arrays.py
```

## 🎯 Usage

### Adding New Problems

1. Create a new file in the appropriate category under `problems/`
2. Implement your solution with type hints and docstrings
3. Add corresponding tests in `tests/`
4. Run tests to ensure everything works

### Example Problem Structure

```python
"""
Problem: Two Sum
Difficulty: Easy
Category: Arrays
"""

from typing import List, Optional

def two_sum(nums: List[int], target: int) -> Optional[List[int]]:
    """
    Find two numbers in the array that add up to the target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        Indices of the two numbers that add up to target, or None if not found
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None
```

## 📊 Problem Categories

- **Arrays & Hashing**: Basic array operations, hash maps
- **Two Pointers**: Techniques for array traversal
- **Sliding Window**: Substring and subarray problems
- **Stack**: LIFO data structure problems
- **Binary Search**: Search algorithms
- **Linked Lists**: Node-based data structures
- **Trees**: Binary trees, BST, traversals
- **Tries**: Prefix trees for string problems
- **Heap/Priority Queue**: Priority-based problems
- **Backtracking**: Recursive exploration
- **Graphs**: Graph traversal and algorithms
- **Advanced Graphs**: Union find, topological sort
- **1-D Dynamic Programming**: Linear DP problems
- **2-D Dynamic Programming**: Matrix DP problems
- **Greedy**: Optimal choice algorithms
- **Intervals**: Overlapping intervals problems
- **Math & Geometry**: Mathematical problems
- **Bit Manipulation**: Binary operations

## 🔧 Development Tools

- **Python 3.12+**: Latest Python features
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Fast Python linter
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality
- **coverage**: Test coverage reporting

## 📈 Progress Tracking

Track your progress by updating the problem list in `docs/progress.md`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your solution with tests
4. Ensure all tests pass and code is formatted
5. Submit a pull request

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.
