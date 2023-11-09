import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import boto3
import botocore
import torch
from label_studio_ml.model import LabelStudioMLBase

# Matches the S3 filepath format given by Label Studio
FILEPATH_REGEX = re.compile(r"^s3://([^/]+)/(.+)$")


class Yolov5Model(LabelStudioMLBase):
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
        super(Yolov5Model, self).__init__(project_id, **kwargs)

        # Fail fast if these aren't set
        domain = os.environ["CLOUD_STORAGE_DOMAIN"]
        region = os.environ["CLOUD_STORAGE_REGION"]
        key = os.environ["CLOUD_STORAGE_KEY"]
        secret = os.environ["CLOUD_STORAGE_SECRET"]
        model_path = os.environ["MODEL_PATH"]

        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.0))
        yolov5_path = os.getenv("YOLOV5_PATH", ".")

        self.model = torch.hub.load(yolov5_path, "custom", model_path, source="local")
        self.model_name = Path(model_path).stem
        self.bucket = os.getenv("CLOUD_STORAGE_BUCKET")

        session = boto3.session.Session()
        self.client = session.client(
            "s3",
            config=botocore.config.Config(
                s3={"addressing_style": "virtual"}
            ),  # Configures to use subdomain/virtual calling format.
            region_name=region,
            endpoint_url=f"https://{region}.{domain}",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
        )

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """
        Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
        :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        predictions = []
        for task in tasks:
            # Prepare to download the image
            if task["storage_filename"]:
                bucket = self.bucket
                filename = task["storage_filename"]
            else:
                cloud_storage_path = task["data"]["image"]
                (bucket, filename) = FILEPATH_REGEX.match(cloud_storage_path).groups()
            image = f"/tmp/image"

            try:
                # Download the image and run the model on it
                self.client.download_file(bucket, filename, image)
                model_results = self.model(image)

                # Get image dimensions from the tensor; this is needed for the bounding box conversions below
                height = model_results.ims[0].shape[0]
                width = model_results.ims[0].shape[1]

                # Iterate through the results returned by the model and format each one for Label Studio
                results = []
                for result in model_results.pandas().xyxy[0].to_dict(orient="records"):
                    if result["confidence"] > self.confidence_threshold:
                        results.append(
                            {
                                # from_name and to_name come from the Label Studio "Labeling Interface" names; it is critical they match the Label Studio values
                                "from_name": "label",
                                "to_name": "image",
                                "type": "rectanglelabels",  # should match the "to" type in the "Labeling Interface"
                                "value": {
                                    "rectanglelabels": [
                                        result["name"]
                                    ],  # Currently copying from the result name value but this can be anything
                                    # Label Studio requires bounding box dimensions to be expressed as percentages of the image
                                    "x": result["xmin"] / width * 100.0,
                                    "y": result["ymin"] / height * 100.0,
                                    "width": (result["xmax"] - result["xmin"])
                                    / width
                                    * 100.0,
                                    "height": (result["ymax"] - result["ymin"])
                                    / height
                                    * 100.0,
                                },
                                "score": result["confidence"],
                            }
                        )

                predictions.append(
                    {"result": results, "model_version": self.model_name}
                )
            finally:
                # Delete the image
                Path(image).unlink(missing_ok=True)

        return predictions

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
