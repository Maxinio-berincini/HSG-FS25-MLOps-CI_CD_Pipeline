import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.data_utils import estimate_required_samples, calculate_achievable_confidence


class TestStatisticalEvaluation(unittest.TestCase):
    
    def test_estimate_required_samples_basic(self):
        """Test that estimate_required_samples returns expected values for known inputs."""
        # Test with standard values
        self.assertEqual(estimate_required_samples(0.1, 0.05), 185)  # Approximate value
        
    def test_estimate_required_samples_edge_cases(self):
        """Test estimate_required_samples with edge cases."""
        # Very small epsilon should require many samples
        small_epsilon = 0.01
        normal_delta = 0.05
        large_samples = estimate_required_samples(small_epsilon, normal_delta)
        self.assertGreater(large_samples, 10000)
        
        # Very small delta (high confidence) should require more samples
        normal_epsilon = 0.1
        small_delta = 0.001
        high_confidence_samples = estimate_required_samples(normal_epsilon, small_delta)
        self.assertGreater(high_confidence_samples, 
                          estimate_required_samples(normal_epsilon, 0.05))
    
    def test_estimate_required_samples_monotonicity(self):
        """Test that smaller epsilon or delta values require more samples."""
        # Decreasing epsilon should increase required samples
        epsilon1, epsilon2 = 0.1, 0.05
        delta = 0.05
        self.assertGreater(estimate_required_samples(epsilon2, delta),
                          estimate_required_samples(epsilon1, delta))
        
        # Decreasing delta should increase required samples
        epsilon = 0.1
        delta1, delta2 = 0.05, 0.01
        self.assertGreater(estimate_required_samples(epsilon, delta2),
                          estimate_required_samples(epsilon, delta1))
    

    def test_estimate_required_samples_formula(self):
        """Test that the function follows Hoeffding's inequality formula."""
        epsilon = 0.1
        delta = 0.05
        expected = int(np.ceil((np.log(2/delta)) / (2 * epsilon**2)))
        self.assertEqual(estimate_required_samples(epsilon, delta), expected)
    
    def test_calculate_achievable_confidence_basic(self):
        """Test claculate_achievable_confidence returns expected values for known inputs."""
        test_size = 10000
        epsilon = 0.02
        confidence = calculate_achievable_confidence(test_size, epsilon)
        self.assertAlmostEqual(confidence, 0.999, places=2)  # Expected value with these parameters
    
    def test_calculate_achievable_confidence_bounds(self):
        """Test that confidence is properly bounded between 0 and 1."""
        # Normal case
        self.assertLessEqual(calculate_achievable_confidence(100, 0.1), 1.0)
        self.assertGreaterEqual(calculate_achievable_confidence(100, 0.1), 0.0)
        
        # Edge case with extremely high confidence (could exceed 1 without bounding)
        self.assertEqual(calculate_achievable_confidence(10000, 0.5), 1.0)
        
        # Edge case with extremely low confidence (could be negative without bounding)
        self.assertEqual(calculate_achievable_confidence(1, 0.01), 0.0)
    
    def test_calculate_achievable_confidence_monotonicity(self):
        """Test that larger test sets or larger epsilon values increase confidence."""
        epsilon = 0.1
        
        # Increasing test set size should increase confidence
        test_size1, test_size2 = 100, 200
        self.assertLess(calculate_achievable_confidence(test_size1, epsilon),
                       calculate_achievable_confidence(test_size2, epsilon))
        
        # Increasing epsilon should increase confidence
        test_size = 100
        epsilon1, epsilon2 = 0.1, 0.2
        self.assertLess(calculate_achievable_confidence(test_size, epsilon1),
                       calculate_achievable_confidence(test_size, epsilon2))
    
    def test_calculate_achievable_confidence_formula(self):
        """Test that the function correctly implements the formula."""
        test_size = 100
        epsilon = 0.1
        
        # Manual calculation
        delta = 2 * np.exp(-2 * test_size * epsilon**2)
        expected = min(max(1 - delta, 0.0), 1.0)
        
        self.assertAlmostEqual(calculate_achievable_confidence(test_size, epsilon), expected)
    
    def test_functions_relationship(self):
        """Test the relationship between the two functions."""
        epsilon = 0.1
        delta = 0.05
        confidence = 1 - delta
        
        # If we calculate the required samples for epsilon and delta,
        # then the achievable confidence with that many samples should be >= the target confidence
        required_samples = estimate_required_samples(epsilon, delta)
        achieved_confidence = calculate_achievable_confidence(required_samples, epsilon)
        
        self.assertGreaterEqual(achieved_confidence, confidence)

if __name__ == '__main__':
    unittest.main()