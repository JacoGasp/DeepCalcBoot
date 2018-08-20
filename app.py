import json
from mathematics import *
import telepot
from telepot.loop import MessageLoop
from handwriting_recognition import *
import time
import os
import logging
import yaml
from logging.config import dictConfig

dir_path = os.path.dirname(os.path.realpath(__file__))

should_listen = True
bot = None

# Load tokens
with open(os.path.join(dir_path, "tokens.yaml"), "rt") as token_file:
    tokens = yaml.safe_load(token_file.read())


def evaluate_math_expression_from_image(image_file):
    results = query_cognitive_vision(image_file)
    string_expression = extract_text(results)
    evaluation = calculate_result(string_expression)

    return evaluation


def on_messaged_arrived(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    logger.info("Message arrived from: {}".format(chat_id))
    logger.debug("Message info: Content Type: {}; Chat Type: {}; Chat Id: {}".format(content_type, chat_type, chat_id))

    logger_msg.info(json.dumps(msg, indent=2))

    if content_type != 'photo':
        logger.info("Message is not image but type {}".format(content_type))
        bot.sendMessage(chat_id, "Please send me an image containing an arithmetical expression.")

    if content_type == 'photo':
        bot.sendMessage(chat_id, "I'm doing the math")
        file_id = msg["photo"][-1]["file_id"]

        file_path = os.path.join("downloads", file_id)

        if not os.path.isfile(file_path):
            logger.debug("Downloading file...")
            bot.download_file(file_id, "downloads/" + file_id)
            logger.debug("Done.")

        image_file = open(os.path.join(dir_path, file_path), "rb").read()
        evaluation = evaluate_math_expression_from_image(image_file)

        if "result" in evaluation:
            bot.sendMessage(chat_id, "Result:\n{} = {:.2f}".format(evaluation["expression"], evaluation["result"]))
        else:
            message = "Something is wrong with your expression: *{}*\n{} ".format(evaluation["expression"],
                                                                                  evaluation["error"])
            bot.sendMessage(chat_id, message, parse_mode="markdown")
            logger.debug("Wrong equation: " + evaluation["error"])


def start_listening_bot():
    MessageLoop(bot, on_messaged_arrived).run_as_thread()
    while should_listen:
        time.sleep(10)


if __name__ == "__main__":
    logger = logging.getLogger('DeepCalculatorBot')
    logger_msg = logging.getLogger('DeepCalculatorBotMsg')

    # create the logger
    with open(os.path.join(dir_path, 'logging.yaml'), 'rt') as f:
        config = yaml.safe_load(f.read())
    dictConfig(config)

    try:
        bot = telepot.Bot(tokens["telegram_bot_key"])

        logger.info("Good morning! I'm now listening to new messages.\n")
        start_listening_bot()

    except KeyboardInterrupt:
        should_listen = False
        logger.info("Closing DeepCalculatorBot. Good night, Bye\n")
