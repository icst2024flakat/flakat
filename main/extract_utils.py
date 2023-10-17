import os
import javalang

cols = [
    'id',
    'project_url',
    'raw',
    'category',
    'is_flaky',
]

def read_test_file_content(test_file_path):
    """
    Returns contents of specified file as a string
    """
    if os.path.exists(test_file_path) == False:
        print("Missing file: {}".format(test_file_path))
        return None
    with open(test_file_path, "r") as tf:
        raw_file_content = tf.read()
    return raw_file_content


def extract_raw_test_methods(raw_test_content):
    """
    Takes input a string with the contents of a .java test file.
    Returns a dictionary of all test method contents in the file.
    (key: unit test method name, value: unit test method contents)
    """
    if raw_test_content == None or len(raw_test_content) == 0:
        return {}
    
    tokens = list(javalang.tokenizer.tokenize(raw_test_content))
    raw_test_methods = {}
    current_raw_test = []
    num_tokens = len(tokens)
    i = 0
    while i < num_tokens:
        # todo: check for test cases without @Test annotations
        if isinstance(tokens[i], javalang.tokenizer.Annotation):
            # check if next token value is "@Test" or "@Deployment"
            if ('Test' == tokens[i+1].value or 'Deployment' == tokens[i+1].value) and isinstance(tokens[i+1], javalang.tokenizer.Identifier):
                # loop through method annotation until signature start
                while i < num_tokens and tokens[i].value != 'void':
                    current_raw_test.append(tokens[i].value)
                    if tokens[i].value != '@':
                        current_raw_test.append(' ')
                    i += 1
                current_test_name = ''
                # loop through method header until '{'
                while i < num_tokens and tokens[i].value != '{':
                    # add current token to output
                    current_raw_test.append(tokens[i].value)
                    if not (tokens[i+1].value == '(' or tokens[i+1].value == ')'):
                        current_raw_test.append(' ')
                    if len(current_test_name) == 0 and isinstance(tokens[i], 
                        javalang.tokenizer.Identifier):
                        current_test_name = tokens[i].value
                    i += 1
                # acount for opening '{' for method body
                current_raw_test.append(tokens[i].value)
                # current_raw_test.append('\n')
                i += 1

                # add current raw test method content to results 
                raw, i = extract_method_body(tokens, i, current_raw_test)
                raw_test_methods[current_test_name] = raw + '}'
                current_raw_test.clear()
        # check for when test's do not have @Test annotation
        elif isinstance(tokens[i], javalang.tokenizer.Keyword) and 'void' == tokens[i].value:
            if isinstance(tokens[i+1], javalang.tokenizer.Identifier) and 'test' in tokens[i+1].value.lower():
                current_test_name = tokens[i+1].value
                # loop through method header until '{'
                while i < num_tokens and tokens[i].value != '{':
                    current_raw_test.append(tokens[i].value)
                    if not (tokens[i+1].value == '(' or tokens[i+1].value == ')'):
                        current_raw_test.append(' ')
                    i += 1
                # acount for opening '{' for method body
                current_raw_test.append(tokens[i].value)
                i += 1
                # add current raw test method content to results 
                raw, i = extract_method_body(tokens, i, current_raw_test)
                raw_test_methods[current_test_name] = raw + '}'
                current_raw_test.clear()
        i += 1
    return raw_test_methods


def extract_method_body(tokens, start_index, current_raw_test):
    unclosed_bracket_count = 1
    num_tokens = len(tokens)
    i = start_index
    # loop through method body
    while i < num_tokens and unclosed_bracket_count > 0:
        if tokens[i].value == '}':
            unclosed_bracket_count -= 1
        elif tokens[i].value == '{':
            unclosed_bracket_count += 1
        
        if unclosed_bracket_count > 0:
            # add current token to output
            current_raw_test.append(tokens[i].value)
            if tokens[i].value == 'case':
                # check for switch-case statements
                if tokens[i+1].value.isdigit() or tokens[i+1].value.isidentifier():
                    current_raw_test.append(' ')
            elif tokens[i].value.isidentifier():
                # check for variable declaration
                if tokens[i+1].value.isidentifier():
                    current_raw_test.append(' ')
            elif tokens[i].value ==']' and tokens[i+1].value.isidentifier():
                # check for arrays
                current_raw_test.append(' ')
            if tokens[i].value == 'assert' \
            and isinstance(tokens[i+1], javalang.tokenizer.DecimalInteger):
                current_raw_test.append(' ')
            if tokens[i].value == '$':
                current_raw_test.insert(len(current_raw_test)-1, ' ')
                current_raw_test.append(' ')
            # elif tokens[i].value in [';', '{', '}']:
            #     current_raw_test.append('\n')
        i += 1

    # add current raw test method content to results 
    raw = ''.join(current_raw_test)
    return raw, i-1
