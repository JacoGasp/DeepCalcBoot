import telepot
from telepot.loop import MessageLoop
import time
from pprint import pprint
from app import logger

bot = telepot.Bot("***REMOVED***")


def on_messaged_arrived(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    logger.debug(content_type, chat_type, chat_id)

    if content_type == 'text':
        bot.sendMessage(chat_id, msg['text'])

    if content_type == 'photo':
        pprint(msg["photo"])
        file_id = msg["photo"][-1]["file_id"]
        file = bot.download_file(file_id, "downloads/" + file_id)


def start_listening_bot():
    MessageLoop(bot, on_messaged_arrived).run_as_thread()
    logger.info('Listening ...')
    while 1:
        time.sleep(10)


start_listening_bot()