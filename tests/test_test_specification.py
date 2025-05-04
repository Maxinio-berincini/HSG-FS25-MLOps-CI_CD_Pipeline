import unittest
from unittest.mock import patch

# Import your module
# Adjust the import as needed based on your project structure
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_specification import TestCondition

class TestConditionClass(unittest.TestCase):
    def test_init_valid_condition(self):
        """Test initialization with valid condition strings."""
        # Test with '>' operator and significance required
        condition = TestCondition("n - o > 0.05 +/- 0.02, True")
        self.assertEqual(condition.condition, "n - o > 0.05 +/- 0.02")
        self.assertEqual(condition.expression, "n - o > 0.05")
        self.assertEqual(condition.specified_epsilon, 0.02)
        self.assertEqual(condition.compare_op, ">")
        self.assertEqual(condition.n_and_o, "n - o")
        self.assertEqual(condition.threshold, 0.05)
        self.assertTrue(condition.require_significance)
        
        # Test with '<' operator and no significance required
        condition = TestCondition("n - o > 0.1 +/- 0.03, False")
        self.assertEqual(condition.condition, "n - o > 0.1 +/- 0.03")
        self.assertEqual(condition.expression, "n - o > 0.1")
        self.assertEqual(condition.specified_epsilon, 0.03)
        self.assertEqual(condition.compare_op, ">")
        self.assertEqual(condition.n_and_o, "n - o")
        self.assertEqual(condition.threshold, 0.1)
        self.assertFalse(condition.require_significance)
        
        # Test with different spacing
        condition = TestCondition("n-o>0.01+/-0.005, True")
        self.assertEqual(condition.condition, "n-o>0.01+/-0.005")
        self.assertEqual(condition.expression, "n-o>0.01")
        self.assertEqual(condition.specified_epsilon, 0.005)
        self.assertEqual(condition.compare_op, ">")
        self.assertEqual(condition.n_and_o, "n-o")
        self.assertEqual(condition.threshold, 0.01)
        self.assertTrue(condition.require_significance)
    
    def test_init_invalid_conditions(self):
        """Test initialization with invalid condition strings."""
        # Missing +/- part
        with self.assertRaises(ValueError):
            TestCondition("n - o > 0.05, True")
        
        # Missing comparison operator
        with self.assertRaises(ValueError):
            TestCondition("n - o = 0.05 +/- 0.02, True")
        
        # Invalid format
        with self.assertRaises(ValueError):
            TestCondition("n > o + 0.05 +/- 0.02, True")
    
    def test_evaluate_significantly_better_meets_threshold(self):
        """Test when challenger is significantly better and meets threshold."""
        # Create test condition with '>' operator and requiring significance
        condition = TestCondition("n - o > 0.05 +/- 0.02, True")
        
        # Create metrics where challenger is significantly better and meets threshold
        challenger_metrics = {
            "mean_accuracy": 0.9,
            "lower_bound": 0.85,
            "upper_bound": 0.95,
            "eval_epsilon": 0.05,
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.8,
            "lower_bound": 0.75,
            "upper_bound": 0.84,  # Note upper_bound < challenger_lower_bound
            "eval_epsilon": 0.04,
            "confidence": 0.95
        }
        
        # Difference = 0.9 - 0.8 = 0.1, which is > 0.05 + max(0.05, 0.02) = 0.1
        # And challenger is significantly better (0.85 > 0.84)
        result, message = condition.evaluate(challenger_metrics, production_metrics)
       
        
        self.assertFalse(result)
        self.assertIn("Condition not satisfied", message)
    
    def test_evaluate_not_significantly_better_but_meets_threshold_no_significance_required(self):
        """Test when challenger meets threshold but is not significantly better, and significance is not required."""
        # Create test condition with '>' operator but not requiring significance
        condition = TestCondition("n - o > 0.05 +/- 0.02, False")
        
        
        # Create metrics where challenger meets threshold but is not significantly better
        challenger_metrics = {
            "mean_accuracy": 0.95,
            "lower_bound": 0.85,
            "upper_bound": 0.95,
            "eval_epsilon": 0.05,
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.8,
            "lower_bound": 0.75,
            "upper_bound": 0.86,  # Note upper_bound > challenger_lower_bound
            "eval_epsilon": 0.06,
            "confidence": 0.95
        }
        
        # Difference = 0.95 - 0.8 = 0.15, which is > 0.05 + max(0.05, 0.02) = 0.1
        # But challenger is not significantly better (0.85 < 0.86)
        # However, significance is not required
        result, message = condition.evaluate(challenger_metrics, production_metrics)
        
        self.assertTrue(result)
        self.assertIn("not statistically significant", message)
    
    def test_evaluate_not_significantly_better_meets_threshold_significance_required(self):
        """Test when challenger meets threshold but is not significantly better, and significance is required."""
        # Create test condition with '>' operator and requiring significance
        condition = TestCondition("n - o > 0.05 +/- 0.02, True")
        
        # Create metrics where challenger meets threshold but is not significantly better
        challenger_metrics = {
            "mean_accuracy": 0.9,
            "lower_bound": 0.85,
            "upper_bound": 0.95,
            "eval_epsilon": 0.05,
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.8,
            "lower_bound": 0.75,
            "upper_bound": 0.86,  # Note upper_bound > challenger_lower_bound
            "eval_epsilon": 0.06,
            "confidence": 0.95
        }
        
        # Difference = 0.9 - 0.8 = 0.1, which is not > 0.05 + max(0.05, 0.02) = 0.1
        # But challenger is not significantly better (0.85 < 0.86)
        # And significance is required
        result, message = condition.evaluate(challenger_metrics, production_metrics)
        
        self.assertFalse(result)
        self.assertIn("Condition not satisfied", message)
    
    def test_evaluate_does_not_meet_threshold(self):
        """Test when challenger does not meet threshold."""
        # Create test condition with '>' operator
        condition = TestCondition("n - o > 0.2 +/- 0.02, True")
        
        # Create metrics where challenger does not meet threshold
        challenger_metrics = {
            "mean_accuracy": 0.9,
            "lower_bound": 0.85,
            "upper_bound": 0.95,
            "eval_epsilon": 0.05,
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.8,
            "lower_bound": 0.75,
            "upper_bound": 0.84,
            "eval_epsilon": 0.04,
            "confidence": 0.95
        }
        
        # Difference = 0.9 - 0.8 = 0.1, which is < 0.2 + max(0.05, 0.02) = 0.25
        result, message = condition.evaluate(challenger_metrics, production_metrics)
        
        self.assertFalse(result)
        self.assertIn("Condition not satisfied", message)
    
    
    
    def test_evaluate_edge_cases(self):
        """Test edge cases for evaluate function."""
        # Edge case: Epsilon in metrics larger than specified epsilon
        condition = TestCondition("n - o > 0.05 +/- 0.02, True")
        
        challenger_metrics = {
            "mean_accuracy": 0.9,
            "lower_bound": 0.8,
            "upper_bound": 1.0,
            "eval_epsilon": 0.1,  # Larger than specified 0.02
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.8,
            "lower_bound": 0.7,
            "upper_bound": 0.9,
            "eval_epsilon": 0.1,
            "confidence": 0.95
        }
        
        # Should use max(0.1, 0.02) = 0.1 for epsilon
        # Difference = 0.9 - 0.8 = 0.1, which is < 0.05 + 0.1 = 0.15
        result, message = condition.evaluate(challenger_metrics, production_metrics)
        
        self.assertFalse(result)
        self.assertIn("Condition not satisfied", message)
    
    def test_evaluate_boundary_conditions(self):
        """Test boundary conditions for the evaluate function."""
        # Test exact boundary condition where difference equals adjusted threshold
        condition = TestCondition("n - o > 0.1 +/- 0.05, True")
        
        # Create metrics where difference is exactly equal to threshold + epsilon
        challenger_metrics = {
            "mean_accuracy": 0.9,
            "lower_bound": 0.85,
            "upper_bound": 0.95,
            "eval_epsilon": 0.05,
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.75,
            "lower_bound": 0.7,
            "upper_bound": 0.8,  # Ensures significance
            "eval_epsilon": 0.05,
            "confidence": 0.95
        }
        
        # Difference = 0.9 - 0.75 = 0.15, which equals 0.1 + 0.05
        # Should not pass as it needs to be strictly greater than threshold + epsilon
        result, message = condition.evaluate(challenger_metrics, production_metrics)
        
        # Check your implementation to ensure it requires strictly greater
        # Your current code uses '>' so this should fail
        self.assertFalse(result)
        
        # Test with difference just slightly more than threshold + epsilon
        production_metrics["mean_accuracy"] = 0.7499
        result, message = condition.evaluate(challenger_metrics, production_metrics)
        self.assertTrue(result)
        self.assertIn("with statistical significance", message)
    
    def test_evaluate_with_metrics_edge_values(self):
        """Test the evaluate function with metrics having edge values."""
        condition = TestCondition("n - o > 0.05 +/- 0.02, True")
        
        # Test with very high values
        challenger_metrics = {
            "mean_accuracy": 0.99,
            "lower_bound": 0.98,
            "upper_bound": 1.0,
            "eval_epsilon": 0.01,
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.94,
            "lower_bound": 0.93,
            "upper_bound": 0.95,
            "eval_epsilon": 0.01,
            "confidence": 0.95
        }
        
        result, message = condition.evaluate(challenger_metrics, production_metrics)
        self.assertFalse(result)
        
        # Test with very low values
        challenger_metrics = {
            "mean_accuracy": 0.1,
            "lower_bound": 0.08,
            "upper_bound": 0.12,
            "eval_epsilon": 0.02,
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.01,
            "eval_epsilon": 0.005,
            "confidence": 0.95
        }
        
        result, message = condition.evaluate(challenger_metrics, production_metrics)
        self.assertTrue(result)
    
    def test_evaluate_with_missing_metrics_keys(self):
        """Test that evaluate handles missing keys in metrics dictionaries."""
        condition = TestCondition("n - o > 0.05 +/- 0.02, True")
        
        # Missing mean_accuracy
        challenger_metrics = {
            "lower_bound": 0.85,
            "upper_bound": 0.95,
            "eval_epsilon": 0.05,
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.8,
            "lower_bound": 0.75,
            "upper_bound": 0.84,
            "eval_epsilon": 0.04,
            "confidence": 0.95
        }
        
        with self.assertRaises(KeyError):
            condition.evaluate(challenger_metrics, production_metrics)
        
        # Missing bounds
        challenger_metrics = {
            "mean_accuracy": 0.9,
            "eval_epsilon": 0.05,
            "confidence": 0.95
        }
        
        production_metrics = {
            "mean_accuracy": 0.8,
            "lower_bound": 0.75,
            "upper_bound": 0.84,
            "eval_epsilon": 0.04,
            "confidence": 0.95
        }
        
        with self.assertRaises(KeyError):
            condition.evaluate(challenger_metrics, production_metrics)

if __name__ == '__main__':
    unittest.main()
