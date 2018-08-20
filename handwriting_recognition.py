import requests
from app import *

logger = logging.getLogger('DeepCalculatorBot')

vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/"
text_recognition_url = vision_base_url + "RecognizeText"


def query_cognitive_vision(image_data):
    # Note: The request parameter changed for APIv2.
    # For APIv1, it is 'handwriting': 'true'.
    logger.info("Querying Cognitive Services")
    params = {'handwriting': 'true'}
    headers = {'Ocp-Apim-Subscription-Key': tokens["microsoft_cognitive_key"],
               'Content-Type': 'application/octet-stream'}

    response = requests.post(text_recognition_url, headers=headers,
                             params=params, data=image_data)
    # print(response.url)
    response.raise_for_status()

    # The 'analysis' object contains various fields that describe the image. The most
    # relevant caption for the image is obtained from the 'descriptions' property.
    # The recognized text isn't immediately available, so poll to wait for completion.
    import time
    analysis = {}
    while "recognitionResult" not in analysis:
        response_final = requests.get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        time.sleep(1)

    return analysis
