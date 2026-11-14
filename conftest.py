# Empty conftest at the repo root: its presence makes pytest insert the
# checkout root into sys.path, so Plan 02/03 tests import src.ingestion
# modules as namespace packages (tests/test_*.py).
