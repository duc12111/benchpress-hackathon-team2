import json

# Read an jsonl file and convert it into a python list of dictionaries.
def read_jsonl(filename):
    """Reads a jsonl file, adds a 'sample_io' key duplicating 'input_output' if it exists,
    and returns a list of dictionaries."""
    lines = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            json_obj = json.loads(line)
            if 'input_output' in json_obj:
                input_output = json_obj['input_output']
                inputs = input_output.get('inputs', [])
                outputs = input_output.get('outputs', [])
                sample_io = []
                for inp, outp in zip(inputs, outputs):
                    sample_io.append({
                        'input': json.dumps(inp),
                        'output': outp
                    })
                json_obj['sample_io'] = sample_io
            lines.append(json_obj)
    return lines

# Write a python list of dictionaries into a jsonl file
def write_jsonl(filename, lines):
    """Writes a python list of dictionaries into a jsonl file"""
    with open(filename, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line) + "\n")
