"""
Deprecated — use tests.smoke_test_ingestion instead.

    PYTHONPATH=pipeline python -m tests.smoke_test_ingestion
"""

from tests.smoke_test_ingestion import main

if __name__ == "__main__":
    main()
