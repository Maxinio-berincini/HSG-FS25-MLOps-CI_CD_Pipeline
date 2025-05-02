
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
    
        req_sig_str = string_parts[1].strip()
        if not (req_sig_str == "True" or req_sig_str == "False"):
            raise ValueError("Second part of the condition must contain True or False after ,")

        self.require_significance = req_sig_str == "True"


        self.condition = string_parts[0].strip()
        

        parts = self.condition.split('+/-')
        if len(parts) != 2:
            raise ValueError("Condition must be in the form 'n - o > d +/- epsilon'")
        
        self.expression = parts[0].strip()
        self.specified_epsilon = float(parts[1].strip())

        ## extract the values from the expression
        ## currently only support '>' for our simple implementation, could be extendet to support any comparison operator
        if '>' in self.expression:
            self.compare_op = '>'
            expression_parts = self.expression.split('>')
        else: 
            raise ValueError("Condition must contain the comparison operator ('>')")
        
        ## self.n_and_o = expression_parts[0].strip() ## actually never used in this simple implementation with only one condition format
        self.n_and_o = expression_parts[0].strip()
        self.threshold = float(expression_parts[1].strip())


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
            
        
        # Determine result based on significance requirements
        if meets_threshold and is_significantly_better:
            return True, f"Challenger passes condition {self.condition} with statistical significance"
        elif meets_threshold and not is_significantly_better and not self.require_significance:
            return True, f"Challenger passes condition but not statistically significant"
        elif meets_threshold and not is_significantly_better and self.require_significance:
            return False, f"Challenger meets threshold but is not statistically significant"
        
        elif not meets_threshold:
                return False, f"Condition not satisfied for challenger: {self.condition} (difference: {difference:.4f}, threshold: {adjusted_threshold:.4f})"
        else:
            return False, "Unexpected condition state encountered" 
    
    


    def print(self):
            print(f"Condition: {self.condition}")
            print(f"Expression: {self.expression}")
            print(f"Comparison Operator: {self.compare_op}")
            print(f"n and o: {self.n_and_o}")
            print(f"d: {self.threshold}")
            print(f"Epsilon: {self.specified_epsilon}")
            print(f"Condition String: {self.condition}")







