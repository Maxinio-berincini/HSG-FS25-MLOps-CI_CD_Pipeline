
import re

## A class to sepcify test condition in the ease.ML domain specific lannguage
class TestCondition:
    
    ## parse a condition string in the following format:
    ## ' n - o > d +/- episilon ' where
    ## n : accuracy of the new (challenger) model
    ## o : accuracy of the old (production) model
    ## d : minimum accuracy improvement
    ## episilon : the half-width of the confidence interval


    def __init__(self, condition_str):

        string_parts = condition_str.split(',')
    
        self.require_significance = bool(string_parts[1].strip())
        self.condition = string_parts[0].strip()
        

        parts = self.condition.split('+/-')
        if len(parts) != 2:
            raise ValueError("Condition must be in the form 'n - o > d +/- epsilon'")
        
        self.expression = parts[0].strip()
        self.specified_epsilon = float(parts[1].strip())

        ## extract the values from the expression
        if '>' in self.expression:
            self.compare_op = '>'
            expression_parts = self.expression.split('>')
        elif '<' in self.expression:
            self.compare_op = '<'
            expression_parts = self.expression.split('<')
        else: 
            raise ValueError("Condition must contain a comparison operator ('>' or '<')")
        
        ## self.n_and_o = expression_parts[0].strip() ## actually never used in this simple implementation with only one condition format
        self.n_and_o = expression_parts[0].strip()
        self.threshold = expression_parts[1].strip()


    ## Evaluate the condition based on the challenger and production metrics.
    ## Returns (bool, str) where bool indicates if the condition is satisfied and str is a justification message
    def evaluate(self, challenger_metrics, production_metrics):
        """
        Evaluate the condition based on the challenger and production metrics.
        """
        n = challenger_metrics['mean_accuracy']
        o = production_metrics['mean_accuracy']
        
        difference = n - o
        
        n_lower = challenger_metrics['lower_bound']
        n_upper = challenger_metrics['upper_bound']
        o_lower = production_metrics['lower_bound']
        o_upper = production_metrics['upper_bound']

        ## determine if challenger is significantly better
        is_significantly_better = n_lower > o_upper

        # Use the effective threshold based on observed error margins
        actual_epsilon = max(challenger_metrics['eval_epsilon'], self.specified_epsilon)
        
        # Check condition
        if self.compare_op == '>':
            adjusted_threshold = self.threshold + actual_epsilon
            meets_threshold = difference > adjusted_threshold
        else:  # '<'
            meets_threshold = difference < adjusted_threshold
            adjusted_threshold = self.threshold - actual_epsilon
        
        # Determine result based on significance requirements
        if meets_threshold and is_significantly_better:
            return True, f"Challenger passes condition {self.condition} with statistical significance"
        elif meets_threshold and not self.require_significance:
            return True, f"Challenger passes condition but not statistically significant"
        elif meets_threshold and self.require_significance:
                return False, f"Challenger meets threshold"
        

        return False, "Condition evaluation failed"
    
    


    def print(self):
            print(f"Condition: {self.condition}")
            print(f"Expression: {self.expression}")
            print(f"Comparison Operator: {self.compare_op}")
            print(f"n and o: {self.n_and_o}")
            print(f"d: {self.threshold}")
            print(f"Epsilon: {self.specified_epsilon}")
            print(f"Condition String: {self.condition}")







