from py_expression_eval import Parser
import logging

logger = logging.getLogger('DeepCalculatorBot')


def extract_text(msg):
    try:
        if msg["status"] != "Succeeded":
            raise ValueError

        lines = msg["recognitionResult"]["lines"]
        raw_message = []
        for line in lines:
            raw_message += list(line["text"])

        logger.debug("RAW Message: {}".format("".join(raw_message)))
        return "".join(raw_message)

    except Exception as e:
        logger.exception(e)


def calculate_result(expression):
    try:
        expression = expression.replace(":", "/")
        expression = expression.replace("x", "*")
        expression = expression.replace(" ", "").replace("+", " + ").replace("-", " - ").replace("*", " * ").replace(
            "/", " / ")
        parser = Parser()
        result = parser.parse(expression).evaluate({})
        logger.debug("Result: {:.2f}".format(result))

        return {
            "expression": expression,
            "result": result
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e)
        }
