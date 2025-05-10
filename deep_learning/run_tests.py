import unittest
import os
import sys

def run_tests():
    """Run all tests in the deep learning models"""
    # Get the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Find all test files
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                # Get the module path
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]
                
                # Add the directory to the Python path
                sys.path.append(root)
                
                # Import the test module
                test_module = __import__(module_name)
                
                # Add the tests to the suite
                test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(test_module))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

if __name__ == '__main__':
    run_tests() 