
from telegram.ext import Updater
import io

class FulautBot():

    def __init__(self):
        self._updater = Updater("604496013:AAEbRwxlcL2L6S7F_oTk1k25iZSAJ4plgDY",
                request_kwargs={'proxy_url':'socks5://127.0.0.1:9050',
                    'urllib3_proxy_kwargs': {"username":"", "password":""}})
    def launch(self):
        self._updater.start_polling()

    def send_image(self, recipient_id, image_path, caption):
        with open(image_path, "rb") as f:
            imageBinaryStream = io.BytesIO(f.read())

        self._updater.bot.send_photo(recipient_id,
                                    imageBinaryStream, caption=caption)
