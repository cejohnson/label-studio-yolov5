import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import boto3
import botocore
import torch
from label_studio_ml.model import LabelStudioMLBase

# Fail fast if these aren't set
SPACES_DOMAIN = os.environ['SPACES_DOMAIN']
SPACES_REGION = os.environ['SPACES_REGION']
SPACES_KEY = os.environ['SPACES_KEY']
SPACES_SECRET = os.environ['SPACES_SECRET']
MODEL = os.environ['MODEL']

CONFIDENCE_THRESHOLD = os.getenv('CONFIDENCE_THRESHOLD', 0.0)

# Matches the spaces filepath format given by Label Studio
FILEPATH_REGEX = re.compile(r'^s3://([^/]+)/(.+)$')

class TreeYolov5Model(LabelStudioMLBase):
    """
    A wrapper around PyTorch/YOLOv5 to allow Label Studio interoperability.

    See:
    https://labelstud.io/guide/ml_create
    https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples
    https://github.com/ultralytics/yolov5
    """

    def __init__(self, project_id: str, **kwargs):
        """
        Initialize the model and the cloud storage client
        """
        super(TreeYolov5Model, self).__init__(project_id, **kwargs)

        self.model = torch.hub.load('.', 'custom', MODEL, source='local')
        
        session = boto3.session.Session()
        self.client = session.client('s3',
                        config=botocore.config.Config(s3={'addressing_style': 'virtual'}), # Configures to use subdomain/virtual calling format.
                        region_name=SPACES_REGION,
                        endpoint_url=F'https://{SPACES_REGION}.{SPACES_DOMAIN}',
                        aws_access_key_id=SPACES_KEY,
                        aws_secret_access_key=SPACES_SECRET)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ 
        Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
        :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        assert len(tasks) == 1
        task = tasks[0]

        # Prepare to download the image
        filepath = task['data']['image']
        (bucket, filename) = FILEPATH_REGEX.match(filepath).groups()
        image = f'/tmp/{filename}'

        try:
            # Download the image and run the model on it
            self.client.download_file(bucket, filename, image)
            model_results = self.model(image)

            # Get image dimensions from the tensor; this is needed for the bounding box conversions below
            height = model_results.ims[0].shape[0]
            width = model_results.ims[0].shape[1]

            # Iterate through the results returned by the model and format each one for Label Studio
            results = []
            for result in model_results.pandas().xyxy[0].to_dict(orient='records'):
                if result['confidence'] > CONFIDENCE_THRESHOLD:
                    results.append({
                        # from_name and to_name come from the Label Studio "Labeling Interface" names; it is critical they match the Label Studio values
                        'from_name': 'label',
                        'to_name': 'image',
                        'type': 'rectanglelabels', # should match the "to" type in the "Labeling Interface"
                        'value': {
                            'rectanglelabels': [result['name']], # Currently copying from the result name value but this can be anything
                            # Label Studio requires bounding box dimensions to be expressed as percentages of the image
                            'x': result['xmin'] / width * 100.0,
                            'y': result['ymin'] / height * 100.0,
                            'width': (result['xmax'] - result['xmin']) / width * 100.0,
                            'height': (result['ymax'] - result['ymin']) / height * 100.0,
                        },
                        'score': result['confidence']
                    })

            return [{'result': results, 'model_version': 'tree-yolov5s-oct1623'}]
        finally:
            # Delete the image
            Path.unlink(image, missing_ok=True)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        # Leaving this default code commented out for reference if someone wants to implement this in the future

        # use cache to retrieve the data from the previous fit() runs
        # old_data = self.get('my_data')
        # old_model_version = self.get('model_version')
        # print(f'Old data: {old_data}')
        # print(f'Old model version: {old_model_version}')

        # # store new data to the cache
        # self.set('my_data', 'my_new_data_value')
        # self.set('model_version', 'my_new_model_version')
        # print(f'New data: {self.get("my_data")}')
        # print(f'New model version: {self.get("model_version")}')

        # print('fit() completed successfully.')
        pass
