from py_expression_eval import Parser
# import app

def extract_symbols_from_text(msg):

    try:
        if msg["status"] != "Succeeded":
            raise ValueError

        lines = msg["recognitionResult"]["lines"]

        str_numbers = []
        for line in lines:
            str_numbers += list(line["text"])
        str_numbers = [x for x in str_numbers if x in "1234567890.+-*/()"]

        i = 0
        while not str_numbers[i].isdigit():
            del(str_numbers[i])
            break

        return str_numbers

    except Exception as e:
        print()
        # app.logger.exception(e)


def numbers_to_expression(symbols):
    i = 0
    expression = []
    clusters = [[]]
    j = 0
    for s in symbols:
        if s.isdigit():
            clusters[j].append(s)
        else:
            j += 2
            clusters.append(s)
            if symbols.index(s) < len(symbols) - 1:
                clusters.append([])

    for cluster in clusters:
        if len(cluster) > 1:
            n = "".join(cluster)
            expression.append(int(n))
        elif cluster[0].isdigit():
            expression.append(int(cluster[0]))
        else:
            expression.append(cluster)
    if type(expression[-1]) != int and not expression[-1].isdigit():
        expression.pop(-1)

    return expression


def calculate_result(expression):
    parser = Parser()
    str_expression = "".join(str(x) for x in expression)
    result = parser.parse(str_expression).evaluate({})
    return {
        "expression": str_expression,
        "result": result
    }
