"""Tests for module hello-world functions and initial health checks."""

import unittest

from accidents.hello import hello_accidents
from app.hello import hello_app
from detection.hello import hello_detection
from evaluation.hello import hello_evaluation
from fuzzy.hello import hello_fuzzy
from hazard_map.hello import hello_hazard_map
from routing.hello import hello_routing
from survey.hello import hello_survey


class TestHelloModules(unittest.TestCase):
    def test_hello_detection(self):
        msg = hello_detection()
        self.assertIn("Detection Module", msg)
        self.assertIn("Ready", msg)

    def test_hello_hazard_map(self):
        msg = hello_hazard_map()
        self.assertIn("Hazard Map Module", msg)
        self.assertIn("Ready", msg)

    def test_hello_accidents(self):
        msg = hello_accidents()
        self.assertIn("Accidents Module", msg)
        self.assertIn("Ready", msg)

    def test_hello_fuzzy(self):
        msg = hello_fuzzy()
        self.assertIn("Fuzzy Safety Engine", msg)
        self.assertIn("Ready", msg)

    def test_hello_survey(self):
        msg = hello_survey()
        self.assertIn("Survey Module", msg)
        self.assertIn("Ready", msg)

    def test_hello_routing(self):
        msg = hello_routing()
        self.assertIn("Routing Module", msg)
        self.assertIn("Ready", msg)

    def test_hello_app(self):
        msg = hello_app()
        self.assertIn("App & UI Module", msg)
        self.assertIn("Ready", msg)

    def test_hello_evaluation(self):
        msg = hello_evaluation()
        self.assertIn("Evaluation Module", msg)
        self.assertIn("Ready", msg)


if __name__ == "__main__":
    unittest.main()
