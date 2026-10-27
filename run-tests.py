"""Run all unit tests (API, model, logging). Uses test-* models and logs, separate from production."""
import sys, unittest

suite = unittest.defaultTestLoader.discover('unittests', pattern='*Tests.py', top_level_dir='.')
sys.exit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
