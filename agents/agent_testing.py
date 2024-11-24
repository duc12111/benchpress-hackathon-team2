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
        # Prepare a local namespace for executing the code
        local_vars = {}
        try:
            exec(self.code, {}, local_vars)
        except Exception as e:
            raise RuntimeError(f"Error executing code: {e}")

        if self.fn_name not in local_vars:
            raise NameError(f"Function '{self.fn_name}' not found in the provided code.")

        fn = local_vars[self.fn_name]
        failures = []
        for idx, (input_args, expected_output_list) in enumerate(zip(self.inputs, self.expected_outputs)):
            expected_output = expected_output_list[0]
            try:
                result = fn(*input_args)
                if not self.outputs_match(result, expected_output):
                    failures.append((idx, input_args, expected_output, result))
            except Exception as e:
                failures.append((idx, input_args, expected_output, str(e)))
        return failures

    def generate_prompt(self, failures):
        prompt = "Please modify your code to also pass the following test cases:\n"
        for idx, input_args, expected_output, actual_output in failures:
            input_str = ', '.join(repr(arg) for arg in input_args)
            prompt += f"\nTest case {idx+1} failed.\n"
            prompt += f"Function call: {self.fn_name}({input_str})\n"
            prompt += f"Expected output: {expected_output}\n"
            prompt += f"Actual output: {actual_output}\n"
        prompt += "\nAll other requirements for robustness, input and output type support, and the other inputs and outputs data shall still be adhered to and be run successfully."
        return prompt
