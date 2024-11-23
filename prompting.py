def generate_prompt(question, fn_name, input_examples, output_examples):
    """
    Generates a robust prompt for code generation dynamically based on inputs.

    Args:
        problem_description (str): A detailed description of the problem.
        fn_name (str): Name of the function to be implemented.
        input_examples (list): A list of input example lists.
        output_examples (list): A list of corresponding output examples.

    Returns:
        str: A complete formatted prompt for the AI to generate Python code.
    """
    # Create the prompt dynamically
    test_data = generate_test_cases(input_examples, output_examples)
    prompt = f"""
As an AI language model, your task is to write a Python function based on the following problem description, function definition, and input/output examples.

# Problem Description:

{question}

# Function Definition:

{fn_name}

## Input and Output Examples:

{test_data}

## Instructions:

Understand the Problem: Carefully read the problem description and analyze the input and output examples to fully understand what is required.

Plan Your Solution: Before coding, plan your approach to solving the problem. Outline the logic, algorithms, or calculations you will use.

Write the Code: Implement your solution as a Python function fully. Assume there are no previous implementations. The code should be robust, efficient, and adhere to best practices.

Output Format: Provide your solution in an XML format that separates the planning from the code, making it easy to parse. Use the following structure:

<solution>
    <planning>
        Provide a detailed planning and explanation here. Describe the steps you will take to solve the problem, the algorithms or calculations used, and any special considerations.
    </planning>
    <code>
<![CDATA[
        # Your complete Python function code here
]]>
    </code>
</solution>
Important Notes:

The code shall consist of only the function definition and its body and include any necessary imports for the functions. Do not include any code for input/output (e.g., no print statements, no input prompts).
Use <![CDATA[ ... ]]> within the <code> tags to ensure that special characters are handled correctly in XML.
Ensure that the XML is well-formed and can be parsed without errors.
Example:

Suppose the problem is:

# Problem Description:

"Given two integers a and b, return their sum."

# Function Definition:

def add_numbers(a, b):

## Input and Output Examples:

Input: "[1, 2]"
Expected output: 3
Input: "[5, 7]"
Expected output: 12
Input: "[-1, -3]"
Expected output: -4

Your response should be:

<solution>
    <planning>
        To solve this problem, I need to create a function that takes two integers as input and returns their sum. I can simply use the `+` operator to add the two numbers.
    </planning>
    <code>
<![CDATA[
def add_numbers(a, b):
    return a + b
]]>
    </code>
</solution>
Important: Your response must be in XML format and contain the Python code inside <code> block to solve the problem defined above.
"""

    return prompt

    
def generate_test_cases(inputs, outputs):
    """
    Generates formatted test cases for inclusion in the prompt.

    Args:
        input_output (dict): A dictionary containing "inputs" and "outputs" keys with corresponding data.

    Returns:
        str: A string containing the formatted test cases.
    """
    # Ensure inputs and outputs have the same length
    if len(inputs) != len(outputs):
        raise ValueError("The number of inputs and outputs must match.")
    
    test_cases = "## Sample Test cases:\n"
    for i, (input_set, output_set) in enumerate(zip(inputs, outputs)):
        test_cases += f"Input:\n{input_set}\nExpected output:\n{output_set[0]}\n"
    
    return test_cases