from py_expression_eval import Parser
import logging

logger = logging.getLogger('DeepCalculatorBot')


def extract_symbols_from_text(msg):

    try:
        if msg["status"] != "Succeeded":
            raise ValueError

        lines = msg["recognitionResult"]["lines"]

        str_numbers = []
        for line in lines:
            str_numbers += list(line["text"])
        str_numbers = [x for x in str_numbers if x in "1234567890.+-*x:÷/()"]

        i = 0
        while not str_numbers[i].isdigit():
            del(str_numbers[i])
            break
        logger.debug(str_numbers)
        return str_numbers

    except Exception as e:
        print()
        logger.exception(e)


def numbers_to_expression(symbols):

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

    logger.debug("Expression: {}".format(expression))
    return expression


def calculate_result(expression):
    parser = Parser()
    str_expression = "".join(str(x) for x in expression)
    result = parser.parse(str_expression).evaluate({})
    logger.debug(result)

    return {
        "expression": str_expression,
        "result": result
    }
