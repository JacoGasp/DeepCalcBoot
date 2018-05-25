from handwriting_recognition import *
from mathematics import *
from pprint import pprint
import json

# image_path = "/Users/chioma/Desktop/img3.jpg"
# image_data = open(image_path, "rb").read()
#
# results = query_cognitive_vision(image_data)
#
# with open('results.json', 'w') as file:
#     json.dump(results, file)

with open('results.json', 'r') as file:
    results = json.load(file)

print(json.dumps(results, indent=2))

str_numbers = extract_symbols_from_text(results)
print("str_numbers", str_numbers)
expression = numbers_to_expression(str_numbers)
print("Expression:", expression)
result = calculate_result(expression)
print("Result:", result)