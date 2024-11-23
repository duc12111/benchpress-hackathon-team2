from .Dataset import Dataset
from evaluations.evalute import contest_evaluate, contest_evaluate_public_tests
from constants.paths import *

class CodeWarsDataset(Dataset):
    def __init__(
        self,
        path: str = CODEWARS_DATA_PATH,
    ):
        super().__init__(path)
        self.id_key = "problem_id"

    def evaluate(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        # Collect tests from 'input_output' field
        tests = []
        inputs = item["input_output"]["inputs"]
        outputs = item["input_output"]["outputs"]
        for inp, outp in zip(inputs, outputs):
            tests.append({
                "input": inp,
                "output": outp
            })

        # Include 'test_cases' if present
        if "test_cases" in item and "inputs" in item["test_cases"]:
            test_cases_inputs = item["test_cases"]["inputs"]
            test_cases_outputs = item["test_cases"]["outputs"]
            for inp, outp in zip(test_cases_inputs, test_cases_outputs):
                tests.append({
                    "input": inp,
                    "output": outp
                })

        return contest_evaluate(
            generated_code=cur_imp,
            id=item[self.id_key],
            tests=tests,
            lang=language
        )

    def evaluate_sample_io(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        # Use the first input/output pair as the sample I/O
        if len(item["input_output"]["inputs"]) == 0:
            return True, ""
        sample_inputs = item["input_output"]["inputs"][:1]
        sample_outputs = item["input_output"]["outputs"][:1]
        tests = []
        for inp, outp in zip(sample_inputs, sample_outputs):
            tests.append({
                "input": inp,
                "output": outp
            })
        return contest_evaluate_public_tests(
            generated_code=cur_imp,
            id=item[self.id_key],
            tests=tests,
            lang=language
        )

    @staticmethod
    def get_prompt(item):
        # Construct the prompt from 'question' field
        prompt = f"{item['question']}\n\n"

        # Include a sample input/output if available
        if len(item["input_output"]["inputs"]) > 0:
            sample_input = item["input_output"]["inputs"][0]
            sample_output = item["input_output"]["outputs"][0]
            prompt += f"Sample Input:\n{sample_input}\nSample Output:\n{sample_output}\n\n"

        prompt += "Important: You must follow the input/output format. Input should be taken from standard input and output should be given to standard output.\nNote: If you are writing a function, then after the function definition, take input using the input() function, call the function with specified parameters, and finally print the output of the function."
        return prompt
