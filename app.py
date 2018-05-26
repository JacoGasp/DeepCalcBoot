import json

from handwriting_recognition import *
from mathematics import *
import telepot
from telepot.loop import MessageLoop
import time
import os

import logging
import yaml
from logging.config import dictConfig
logger = logging.getLogger('DeepCalculatorBot')
logger_msg = logging.getLogger('DeepCalculatorBotMsg')

# create the logger
with open('logging.yaml', 'rt') as f:
    config = yaml.safe_load(f.read())
dictConfig(config)

should_listen = True
bot = telepot.Bot("***REMOVED***")


def evaluate_math_expresion_from_image(image_file):
    print("Query online Cognitive Services")
    results = query_cognitive_vision(image_file)

    # with open('results.json', 'w') as file:
    #     json.dump(results, file)
    #
    # with open('results.json', 'r') as file:
    #     results = json.load(file)
    #
    # print(json.dumps(results, indent=2))

    str_numbers = extract_symbols_from_text(results)
    logger.debug("str_numbers", str_numbers)
    expression = numbers_to_expression(str_numbers)
    logger.debug("Expression:", expression)
    evaluation = calculate_result(expression)
    # print("Result", "{} = {}".format(evaluation["expression"], evaluation["result"]))
    return evaluation


def on_messaged_arrived(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    # print(content_type, chat_type, chat_id)
    logger.info("Message arrived")
    logger_msg.info(json.dumps(msg, indent=2))

    if content_type != 'photo':
        bot.sendMessage(chat_id, "Please send me an image containing an arithmetical expression.")

    if content_type == 'photo':
        bot.sendMessage(chat_id, "I'm doing the math")
        file_id = msg["photo"][-1]["file_id"]
        # ("file_id:", file_id)

        file_path = os.path.join("downloads", file_id)

        if not os.path.isfile(file_path):
            logger.debug("Downloading file...", end=" ")
            bot.download_file(file_id, "downloads/" + file_id)
            logger.debug("Done.")

        image_file = open(file_path, "rb").read()
        try:
            evaluation = evaluate_math_expresion_from_image(image_file)
            bot.sendMessage(chat_id, "Result:\n {} = {:.2f}".format(evaluation["expression"], evaluation["result"]))
        except Exception as e:
            bot.sendMessage(chat_id, "I'm sorry but I cannot understand your writing. Put much effort in it, please!")
            logger.exception(e)


def start_listening_bot():
    MessageLoop(bot, on_messaged_arrived).run_as_thread()
    logger.info('Listening ...')
    while should_listen:
        time.sleep(10)


if __name__ == "__main__":
    try:
        logger.info("Started.")
        start_listening_bot()

    except KeyboardInterrupt:
        should_listen = False
        logger.info("Closing DeepCalculatorBot. Bye")
