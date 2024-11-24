import os
import time
import random
import argparse
from typing import List, Dict, Any
from aleph_alpha_client import Client, CompletionRequest, Prompt
from utilities import load_sample, run_test_cases, score
from utils import get_cot_prompt
import json

secrets_path = os.path.join(os.path.dirname(__file__), 'secrets.json')
with open(secrets_path) as f:
    secrets = json.load(f)
    AA_TOKEN = secrets.get("AA_TOKEN")
    
MODEL = "llama-3.1-70b-instruct-long-context"

if not AA_TOKEN:
    raise ValueError("Aleph Alpha Playground token is not set.")

client = Client(AA_TOKEN)

def is_syntax_correct(code: str) -> bool:
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

# def generate_code_solutions(problem: dict, client: Client, num_samples: int) -> List[str]:
#     """
#     Generate multiple code solutions for a given problem.

#     Args:
#         problem (dict): The problem to solve.
#         client (Client): The Aleph Alpha client.
#         num_samples (int): Number of code solutions to generate.

#     Returns:
#         List[str]: A list of generated code solutions.
#     """
#     prompt = generate_code_prompt(problem)

#     code_solutions = []
#     for _ in range(num_samples):
#         request = CompletionRequest(
#             prompt=Prompt.from_text(prompt),
#             maximum_tokens=256,
#             temperature=0.8,  # Higher temperature for diversity
#             stop_sequences=["\n\n"],
#             echo=False
#         )
#         response = client.complete(request, model=MODEL)
#         code = response.completions[0].completion.strip()
#         code_solutions.append(code)
#         # Sleep to respect rate limits
#         time.sleep(0.5)
#     return code_solutions

def generate_code_solutions(problem: dict, client: Client, num_samples: int, temperature: float = 0.8, try_limit: int = 2) -> List[str]:
    """
    Generate multiple code solutions for a given problem, ensuring they have correct syntax.

    Args:
        problem (dict): The problem to solve.
        client (Client): The Aleph Alpha client.
        num_samples (int): Number of code solutions to generate.
        temperature (float): Sampling temperature for diversity.

    Returns:
        List[str]: A list of generated code solutions with correct syntax.
    """
    prompt = generate_code_prompt(problem)
    prompt = get_cot_prompt(prompt)
    code_solutions = []
    max_attempts = num_samples * 2  # Allow up to twice the number of samples to account for syntax errors
    attempts = 0
    while len(code_solutions) < num_samples and attempts < max_attempts:
        request = CompletionRequest(
            prompt=Prompt.from_text(prompt),
            maximum_tokens=2048,
            temperature=temperature,  # Adjusted temperature for diversity
            stop_sequences=["\n\n"],
            echo=False
        )
        response = client.complete(request, model=MODEL)
        code = response.completions[0].completion.strip()
        num_try = 0
        while (num_try <= try_limit):
            if is_syntax_correct(code):
                code_solutions.append(code)
                break
            else:
                num_try += 1
        if num_try > try_limit:
            request = CompletionRequest(
            prompt=f"Please debug the current code \code{code}\code for the given prompt {Prompt.from_text(prompt)} Just the write the fixed code.",
            maximum_tokens=1024,
            temperature=temperature,  # Adjusted temperature for diversity
            stop_sequences=["\n\n"],
            echo=False
        )
            print("Discarded code with syntax error.")
        
        attempts += 1

    if not code_solutions:
        # If no code solutions have correct syntax, return an empty list or handle accordingly
        print("No syntactically correct code solutions were generated.")
    return code_solutions

def generate_code_prompt(problem: dict) -> str:
    """
    Generate a prompt for code generation.

    Args:
        problem (dict): The problem to solve.

    Returns:
        str: The prompt for code generation.
    """
    prompt = f"{problem['starter_code']}\n"
    prompt += f"\"\"\"\n{problem['question']}\n\"\"\"\n"
    prompt += "\n# Write your code below\n"
    # print(prompt)
    # exit()
    return prompt

def generate_test_cases(problem: dict, client: Client, num_samples: int) -> List[str]:
    """
    Generate test cases for a given problem.

    Args:
        problem (dict): The problem to generate test cases for.
        client (Client): The Aleph Alpha client.
        num_samples (int): Number of test cases to generate.

    Returns:
        List[str]: A list of test case strings (assert statements).
    """
    prompt = generate_test_case_prompt(problem)

    test_cases = []
    for _ in range(num_samples):
        request = CompletionRequest(
            prompt=Prompt.from_text(prompt),
            maximum_tokens=64,
            temperature=0.8,
            stop_sequences=["\n\n"],
            echo=False
        )
        response = client.complete(request, model=MODEL)
        test_case = response.completions[0].completion.strip()
        if test_case.startswith("assert"):
            test_cases.append(test_case)

    return test_cases

def generate_test_case_prompt(problem: dict) -> str:
    """
    Generate a prompt for test case generation.

    Args:
        problem (dict): The problem to generate test cases for.

    Returns:
        str: The prompt for test case generation.
    """
    prompt = f"{problem['starter_code']}\n"
    prompt += f"\"\"\"\n{problem['question']}\n\"\"\"\n"
    prompt += "\n# Generate test cases for the function above"
    prompt += "\n# Each test case should be in the form of an assert statement."
    fn_name = problem['input_output']['fn_name']
    prompt += f"\n# Example: assert {fn_name}(input) == expected_output"
    prompt += "\n\n# Test cases:\nassert "
    return prompt

def parse_test_case(test_case_str: str, problem: dict) -> Dict[str, Any]:
    """
    Parse a test case string into input and expected output.

    Args:
        test_case_str (str): The test case string.
        problem (dict): The problem information.

    Returns:
        Dict[str, Any]: Parsed test case or None if parsing fails.
    """
    try:
        if not test_case_str.startswith("assert"):
            return None
        # Remove 'assert' and split at '=='
        test_case_str = test_case_str[len("assert"):].strip()
        if '==' not in test_case_str:
            return None
        left, right = test_case_str.split('==', 1)
        left = left.strip()
        right = right.strip()
        fn_name = problem['input_output']['fn_name']
        if left.startswith(f"{fn_name}(") and left.endswith(')'):
            input_str = left[len(f"{fn_name}("):-1]
            return {'inputs': input_str, 'outputs': right, 'fn_name': fn_name}
        else:
            return None
    except Exception:
        return None

# # def generate_code_with_codet(problem: dict, client: Client, additional_data: bool) -> str:
# #     """
# #     Generate code for a problem using the CODET method.

# #     Args:
# #         problem (dict): The problem to solve.
# #         client (Client): The Aleph Alpha client.
# #         additional_data (bool): Whether to generate additional test cases.

# #     Returns:
# #         str: The selected code solution.
# #     """
# #     num_code_samples = 5   # Adjust the number of code solutions as needed

# #     # Generate code solutions
# #     code_solutions = generate_code_solutions(problem, client, num_code_samples)

# #     if additional_data:
# #         # Generate test cases
# #         num_test_case_samples = 5  # Adjust the number of test cases as needed
# #         test_cases_raw = generate_test_cases(problem, client, num_test_case_samples)

# #         # Parse test cases
# #         test_cases = []
# #         for test_case_str in test_cases_raw:
# #             parsed_test_case = parse_test_case(test_case_str, problem)
# #             if parsed_test_case:
# #                 test_cases.append(parsed_test_case)
# #     else:
# #         # Use existing test cases from 'input_output' and 'test_cases'
# #         test_cases = []
# #         # From 'input_output'
# #         input_output = problem.get('input_output', {})
# #         inputs_list = input_output.get('inputs', [])
# #         outputs_list = input_output.get('outputs', [])
# #         fn_name = input_output.get('fn_name', problem.get('starter_code').split('(')[0].strip())
# #         for inputs, outputs in zip(inputs_list, outputs_list):
# #             test_cases.append({
# #                 'inputs': inputs,
# #                 'outputs': outputs,
# #                 'fn_name': fn_name
# #             })
# #         # From 'test_cases' if available
# #         test_cases_dict = problem.get('test_cases', {})
# #         inputs_list = test_cases_dict.get('inputs', [])
# #         outputs_list = test_cases_dict.get('outputs', [])
# #         fn_name = test_cases_dict.get('fn_name', fn_name)
# #         for inputs, outputs in zip(inputs_list, outputs_list):
# #             test_cases.append({
# #                 'inputs': inputs,
# #                 'outputs': outputs,
# #                 'fn_name': fn_name
# #             })

# #     if not test_cases:
# #         # If no valid test cases were available, return the first code solution
# #         return code_solutions[0]

# #     # Evaluate code solutions on test cases
# #     code_solution_scores = []
# #     for code in code_solutions:
# #         # Prepare a problem dict with the test cases
# #         problem_with_test_cases = problem.copy()
# #         problem_with_test_cases['test_cases'] = {
# #             'fn_name': fn_name,
# #             'inputs': [tc['inputs'] for tc in test_cases],
# #             'outputs': [tc['outputs'] for tc in test_cases]
# #         }

# #         try:
# #             # Run test cases using the provided utility function
# #             test_results = run_test_cases(
# #                 problem=problem_with_test_cases,
# #                 generation=code,
# #                 timeout=10
# #             )

# #             # Count the number of passed test cases
# #             num_passed = sum(1 for result in test_results if result['passed'])
# #             code_solution_scores.append((code, num_passed))

# #         except Exception as e:
# #             # If execution fails, consider zero passed tests
# #             code_solution_scores.append((code, 0))

# #     # Select the code solution with the highest number of passed test cases
# #     if not code_solution_scores:
# #         return code_solutions[0]  # Fallback

# #     best_code, _ = max(code_solution_scores, key=lambda x: x[1])
# #     return best_code

# def generate_code_with_codet(problem: dict, client: Client, additional_data: bool) -> str:
#     """
#     Generate code for a problem using the CODET method.

#     Args:
#         problem (dict): The problem to solve.
#         client (Client): The Aleph Alpha client.
#         additional_data (bool): Whether to generate additional test cases.

#     Returns:
#         str: The selected code solution.
#     """
#     num_code_samples = 5   # Adjust the number of code solutions as needed

#     # Generate code solutions
#     code_solutions = generate_code_solutions(problem, client, num_code_samples)

#     if additional_data:
#         # Generate test cases
#         num_test_case_samples = 5  # Adjust the number of test cases as needed
#         test_cases_raw = generate_test_cases(problem, client, num_test_case_samples)

#         # Parse test cases
#         test_cases = []
#         for test_case_str in test_cases_raw:
#             parsed_test_case = parse_test_case(test_case_str, problem)
#             if parsed_test_case:
#                 test_cases.append(parsed_test_case)
#     else:
#         # Use existing test cases from 'input_output' only
#         test_cases = []
#         # From 'input_output'
#         input_output = problem.get('input_output', {})
#         inputs_list = input_output.get('inputs', [])
#         outputs_list = input_output.get('outputs', [])
#         fn_name = input_output.get('fn_name', problem.get('starter_code').split('(')[0].strip())
#         for inputs, outputs in zip(inputs_list, outputs_list):
#             test_cases.append({
#                 'inputs': inputs,
#                 'outputs': outputs,
#                 'fn_name': fn_name
#             })
#         # Do not use 'test_cases' attribute here

#     if not test_cases:
#         # If no valid test cases were available, return the first code solution
#         return code_solutions[0]

#     # Evaluate code solutions on test cases from 'input_output' only
#     code_solution_scores = []
#     for code in code_solutions:
#         # Prepare a problem dict with the test cases
#         problem_with_test_cases = problem.copy()
#         problem_with_test_cases['test_cases'] = {
#             'fn_name': fn_name,
#             'inputs': [tc['inputs'] for tc in test_cases],
#             'outputs': [tc['outputs'] for tc in test_cases]
#         }

#         try:
#             # Run test cases using the provided utility function
#             test_results = run_test_cases(
#                 problem=problem_with_test_cases,
#                 generation=code,
#                 timeout=10
#             )

#             # Count the number of passed test cases
#             num_passed = sum(1 for result in test_results if result['passed'])
#             code_solution_scores.append((code, num_passed))

#         except Exception as e:
#             # If execution fails, consider zero passed tests
#             code_solution_scores.append((code, 0))

#     # Select the code solution with the highest number of passed test cases
#     if not code_solution_scores:
#         return code_solutions[0]  # Fallback

#     best_code, _ = max(code_solution_scores, key=lambda x: x[1])
#     return best_code


def select_best_code_full(code_solutions: List[str], test_cases: List[Dict[str, Any]], problem: dict) -> str:
    """
    Select the best code solution by evaluating all code solutions on all test cases.

    Args:
        code_solutions (List[str]): List of generated code solutions.
        test_cases (List[Dict[str, Any]]): List of parsed test cases.
        problem (dict): The problem information.

    Returns:
        str: The selected code solution.
    """
    fn_name = test_cases[0]['fn_name'] if test_cases else ''
    code_solution_scores = []
    for code in code_solutions:
        # Prepare a problem dict with the test cases
        problem_with_test_cases = problem.copy()
        problem_with_test_cases['test_cases'] = {
            'fn_name': fn_name,
            'inputs': [tc['inputs'] for tc in test_cases],
            'outputs': [tc['outputs'] for tc in test_cases]
        }

        try:
            # Run test cases using the provided utility function
            test_results = run_test_cases(
                problem=problem_with_test_cases,
                generation=code,
                timeout=10
            )

            # Count the number of passed test cases
            num_passed = sum(1 for result in test_results if result['passed'])
            code_solution_scores.append((code, num_passed))

        except Exception as e:
            # If execution fails, consider zero passed tests
            code_solution_scores.append((code, 0))

    # Select the code solution with the highest number of passed test cases
    if not code_solution_scores:
        return code_solutions[0]  # Fallback

    best_code, _ = max(code_solution_scores, key=lambda x: x[1])
    return best_code

def execute_code_on_test_case(code: str, test_case: Dict[str, Any], timeout: int = 5) -> bool:
    """
    Execute a code solution on a single test case.

    Args:
        code (str): The code solution.
        test_case (Dict[str, Any]): The test case.
        timeout (int): Execution timeout.

    Returns:
        bool: True if the code passes the test case, False otherwise.
    """
    problem_with_test_case = {
        'test_cases': {
            'fn_name': test_case['fn_name'],
            'inputs': [test_case['inputs']],
            'outputs': [test_case['outputs']]
        }
    }
    try:
        test_results = run_test_cases(
            problem=problem_with_test_case,
            generation=code,
            timeout=timeout
        )
        return test_results[0]['passed']
    except Exception:
        return False

def select_best_code_ransac(code_solutions: List[str], test_cases: List[Dict[str, Any]], iterations: int = 5) -> str:
    """
    Select the best code solution using the RANSAC algorithm.

    Args:
        code_solutions (List[str]): List of generated code solutions.
        test_cases (List[Dict[str, Any]]): List of parsed test cases.
        iterations (int): Number of iterations for RANSAC.

    Returns:
        str: The selected code solution.
    """
    # Build a list of all possible pairs (code solution, test case)
    pairs = [(i, j) for i in range(len(code_solutions)) for j in range(len(test_cases))]

    best_consensus_set = []
    best_score = 0
    for _ in range(iterations):
        # Randomly select a pair
        code_idx, test_idx = random.choice(pairs)
        code = code_solutions[code_idx]
        test_case = test_cases[test_idx]

        # Execute code on the test case
        if execute_code_on_test_case(code, test_case):
            # Build consensus set
            consensus_codes = []
            consensus_tests = []
            for i, c in enumerate(code_solutions):
                passed_tests = []
                for j, t in enumerate(test_cases):
                    if execute_code_on_test_case(c, t):
                        passed_tests.append(j)
                if test_idx in passed_tests:
                    consensus_codes.append(i)
            for j, t in enumerate(test_cases):
                passed_codes = []
                for i in consensus_codes:
                    c = code_solutions[i]
                    if execute_code_on_test_case(c, t):
                        passed_codes.append(i)
                if len(passed_codes) == len(consensus_codes):
                    consensus_tests.append(j)
            score = len(consensus_codes) * len(consensus_tests)
            if score > best_score:
                best_score = score
                best_consensus_set = consensus_codes

    if best_consensus_set:
        # Return a code solution from the best consensus set
        best_code_idx = best_consensus_set[0]
        return code_solutions[best_code_idx]
    else:
        # Fallback to the first code solution
        return code_solutions[0]


def generate_code_with_codet(problem: dict, client: Client, additional_data: bool, speed_up: bool) -> str:
    """
    Generate code for a problem using the CODET method with optional RANSAC speed-up.

    Args:
        problem (dict): The problem to solve.
        client (Client): The Aleph Alpha client.
        additional_data (bool): Whether to generate additional test cases.
        speed_up (bool): Whether to use RANSAC speed-up.

    Returns:
        str: The selected code solution.
    """
    # Parameters adjusted for SOTA results on the APPS dataset
    num_code_samples = 20  # For other datasets
    num_test_case_samples = 5
    temperature = 0.8

    # Generate code solutions
    code_solutions = generate_code_solutions(problem, client, num_code_samples, temperature)

    if additional_data:
        # Generate test cases
        test_cases_raw = generate_test_cases(problem, client, num_test_case_samples, temperature)

        # Parse test cases
        test_cases = []
        for test_case_str in test_cases_raw:
            parsed_test_case = parse_test_case(test_case_str, problem)
            if parsed_test_case:
                test_cases.append(parsed_test_case)
    else:
        # Use existing test cases from 'input_output' only
        test_cases = []
        # From 'input_output'
        input_output = problem.get('input_output', {})
        inputs_list = input_output.get('inputs', [])
        outputs_list = input_output.get('outputs', [])
        fn_name = input_output.get('fn_name', problem.get('starter_code').split('(')[0].strip())
        for inputs, outputs in zip(inputs_list, outputs_list):
            test_cases.append({
                'inputs': inputs,
                'outputs': outputs,
                'fn_name': fn_name
            })

    if not test_cases:
        # If no valid test cases were available, return the first code solution
        return code_solutions[0]

    # If speed_up is True, use RANSAC algorithm
    if speed_up:
        best_code = select_best_code_ransac(code_solutions, test_cases)
    else:
        # Evaluate code solutions on test cases from 'input_output' only
        best_code = select_best_code_full(code_solutions, test_cases, problem)

    return best_code

def main():
    parser = argparse.ArgumentParser(description='CODET Benchmark Script')
    parser.add_argument('--additional_data', action='store_true', help='Generate additional test data')
    parser.add_argument('--speed_up', action='store_true', help='Use RANSAC algorithm for speed-up')
    args = parser.parse_args()

    passed_problems, passed_test_cases = score(
        generation_func=lambda problem, client: generate_code_with_codet(problem, client, args.additional_data, args.speed_up),
        client=client,
        dataset_path="./data/val",  # Path to your validation set
        length=100,  # Adjust the number of problems to evaluate
    )
    print(f"Passed {passed_problems*100}% of problems")
    print(f"Passed {passed_test_cases*100}% of test cases")

if __name__ == "__main__":
    import multiprocessing
    # Set the start method to 'spawn' to avoid OSError issues
    multiprocessing.set_start_method('spawn')
    main()