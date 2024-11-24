import traceback

class Agent:
    def __init__(self, code, json_data):
        self.code = code
        self.json_data = json_data
        self.fn_name = json_data["input_output"]["fn_name"]
        self.inputs = json_data["input_output"]["inputs"]
        self.expected_outputs = json_data["input_output"]["outputs"]

    def outputs_match(self, actual, expected):
        if isinstance(expected, float) or isinstance(actual, float):
            # Compare with rounding to two decimal places
            return round(actual, 2) == round(expected, 2)
        else:
            return actual == expected

    def run_tests(self):
        # Prepare a namespace for executing the code
        exec_namespace = {}
        try:
            # Execute the code snippet
            exec(self.code, exec_namespace)
        except Exception as e:
            # Capture the stack trace and add to failures
            error_trace = traceback.format_exc()
            failures.append((-1, None, None, str(e), error_trace))
            return failures  # Return early since code execution failed

        if self.fn_name not in exec_namespace:
            # Capture the error and add to failures
            error_message = f"Function '{self.fn_name}' not found in the provided code."
            error_trace = traceback.format_exc()
            failures.append((-1, None, None, error_message, error_trace))
            return failures  # Return early since function is missing

        fn = exec_namespace[self.fn_name]
        failures = []
        for idx, (input_args, expected_output_list) in enumerate(zip(self.inputs, self.expected_outputs)):
            expected_output = expected_output_list[0]
            try:
                result = fn(*input_args)
                if not self.outputs_match(result, expected_output):
                    failures.append((idx, input_args, expected_output, result, None))
            except Exception as e:
                # Capture the stack trace
                error_trace = traceback.format_exc()
                failures.append((idx, input_args, expected_output, str(e), error_trace))
        return failures

    def generate_prompt(self, failures):
        prompt = "I have run your implementation of the solution for the problem above with the sample input and output data. However, it did not produce the correct results. Please see my results of the running of your code below and modify your code to pass them:\n"
        # Check for code execution errors first
        code_execution_error = any(failure[0] == -1 for failure in failures)
        if code_execution_error:
            for idx, input_args, expected_output, actual_output, error_trace in failures:
                if idx == -1:
                    prompt += f"\nAn error occurred during code execution:\n{actual_output}\n{error_trace}\n"
            prompt += f"\nPlease ensure that the function name is defined as '{self.fn_name}'.\n"
        else:
            for idx, input_args, expected_output, actual_output, error_trace in failures:
                input_str = ', '.join(repr(arg) for arg in input_args)
                prompt += f"\nTest case {idx+1} failed.\n"
                prompt += f"Function call: {self.fn_name}({input_str})\n"
                prompt += f"Expected output: {expected_output}\n"
                if error_trace:
                    prompt += f"An exception occurred during execution:\n{error_trace}\n"
                else:
                    prompt += f"Actual output: {actual_output}\n"
        prompt += """\nDo not provide code for special cases, the generated code shall run generally for all testing inputs and calculate correct outputs. All other requirements for robustness, input and output type support, and the other inputs and outputs data shall still be adhered to and be run successfully.
Provide your solution in an XML format that separates the planning from the code, making it easy to parse. Use the following structure:

<solution>
    <planning>
        Provide a detailed planning and explanation here for how you will change the implementation and where the problem was. Describe the steps you will take to solve the problem, the algorithms or calculations used, and any special considerations.
    </planning>
    <code>
<![CDATA[
        # Your complete Python function code here
]]>
    </code>
</solution>
        """
        return prompt