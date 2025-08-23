#!/usr/bin/env python3
"""
Setup script for initializing the development environment.
"""

import subprocess
import sys


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"⏳ {description}...")
    try:
        _ = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up interview preparation environment...")

    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Warning: Not in a virtual environment. Consider creating one:")
        print("   python -m venv venv")
        print("   .\\venv\\Scripts\\activate  # Windows")
        print("   source venv/bin/activate  # macOS/Linux")
        print()

    # Install dependencies
    success = True

    success &= run_command("pip install --upgrade pip", "Upgrading pip")
    success &= run_command("pip install -r requirements.txt", "Installing dependencies")
    success &= run_command("pre-commit install", "Installing pre-commit hooks")

    if success:
        print("\n🎉 Setup completed successfully!")
        print("\n📚 Next steps:")
        print("1. Run tests: pytest")
        print("2. Format code: black .")
        print("3. Lint code: ruff check .")
        print("4. Run a problem: python problems/arrays/two_sum.py")
    else:
        print("\n❌ Setup encountered errors. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
