from handwriting_recognition import *
from mathematics import *
import telepot
from telepot.loop import MessageLoop
import time
from pprint import pprint
import json
import os

# image_path = "/Users/chioma/Desktop/img3.jpg"
# image_data = open(image_path, "rb").read()
#
# results = query_cognitive_vision(image_data)
#
# with open('results.json', 'w') as file:
#     json.dump(results, file)


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
    print("str_numbers", str_numbers)
    expression = numbers_to_expression(str_numbers)
    print("Expression:", expression)
    evaluation = calculate_result(expression)
    print("Result", "{} = {}".format(evaluation["expression"], evaluation["result"]))
    return evaluation


def on_messaged_arrived(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(content_type, chat_type, chat_id)

    if content_type != 'photo':
        bot.sendMessage(chat_id, "Please send me an image containing an arithmetical expression.")

    if content_type == 'photo':
        bot.sendMessage(chat_id, "I'm doing the math")
        file_id = msg["photo"][-1]["file_id"]
        print("file_id:", file_id)

        file_path = os.path.join("downloads", file_id)

        if not os.path.isfile(file_path):
            print("Downloading file...", end=" ")
            bot.download_file(file_id, "downloads/" + file_id)
            print("Done.")

        image_file = open(file_path, "rb").read()
        try:
            evaluation = evaluate_math_expresion_from_image(image_file)
            bot.sendMessage(chat_id, "Result:\n {} = {:.2f}".format(evaluation["expression"], evaluation["result"]))
        except Exception as e:
            bot.sendMessage(chat_id, "I'm sorry but I cannot understand your writing. Put much effort in it, please!")
            print(e)


def start_listening_bot():
    MessageLoop(bot, on_messaged_arrived).run_as_thread()
    print('Listening ...')
    while 1:
        time.sleep(10)


start_listening_bot()
