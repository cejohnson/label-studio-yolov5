import os
import logging

import requests
from dotenv import load_dotenv
from progress.bar import Bar
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from model import Yolov5Model


def create_predictions(base_url, access_token, project_id, view_id):
    """
    Create predictions for a project (and optionally view/tab) in Label Studio.

    The required parameters are straightforward, the optional `view_id` is an optimization that allows using a view/tab that only contains
    tasks that don't already have predictions, or perhaps only outdated predictions (not currently supported in the code). This allows us to
    avoid iterating through tens of thousands of tasks that we don't need to process.

    :param base_url: base Label Studio url (https://example.com)
    :param access_token: Label Studio access token, this can be found in "Account & Settings" in Label Studio under your login
    :param project_id: in the Label Studio url: https://example.com/projects/{project_id}/data?tab={view_id}
    :param view_id: in the Label Studio url: https://example.com/projects/{project_id}/data?tab={view_id} (optional but recommended)
    """
    logging.info("Initializing")
    model = Yolov5Model(project_id)

    # Set urls
    tasks_url = f"{base_url}/api/tasks"
    predict_url = f"{base_url}/api/predictions"

    # Set retry policy for all requests
    retries = Retry(
        total=5,
        backoff_factor=1,
        allowed_methods={"GET", "POST"},
    )

    # Create the session
    session = requests.Session()
    session.headers.update({"Authorization": f"Token {access_token}"})
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # Setup parameters
    page_size = 50

    params = {"page_size": 1, "project": project_id}
    if view_id:
        params["view"] = view_id

    # Get the total task count
    resp = session.get(tasks_url, params=params)
    resp.raise_for_status()
    r = resp.json()

    total_task_count = r["total"]
    params["page_size"] = page_size

    # Create a progress bar for the total task count and start processing.
    # Because the task count might change, this is really just a best effort estimate.
    # Technically this could probably be updated/recreated on each new page, that might be a decent improvement.
    logging.info("Processing tasks")
    with Bar(
        "Predicting",
        max=total_task_count,
        suffix="%(index)d/%(max)d - %(percent).1f%% - Elapsed: %(elapsed_td)s - ETA: %(eta_td)s",
    ) as bar:
        # Run until there are no more tasks
        while True:
            try:
                # Fetch a batch of `page_size` tasks to process
                resp = session.get(tasks_url, params=params, timeout=10)
                resp.raise_for_status()
                tasks = resp.json()["tasks"]

                if len(tasks) == 0:
                    logging.info("Done")
                    break

                # TODO: this should be fairly easy to parallelize with a basic executor, other than it would likely break the progress bar.
                # I'm not sure how much time is spent on network calls vs running the model, so while multiple processes are a clear win
                # it might also make sense to use multiple threads per process for maximum speedup, not that maximum speed is necessary.
                # Shame python is kind of terrible for this kind of parallelism :/
                for task in tasks:
                    task_id = task["id"]
                    logging.info(f"Running inference for task {task_id}")

                    # TODO: instead of just looking at the prediction count, we might want to see if there are any predictions for this model instead.
                    # Conversely we could require a view and just trust that every task needs to be processed
                    if task["total_predictions"] == 0:
                        # Run the model
                        try:
                            [prediction] = model.predict([task])
                        except Exception:
                            logging.exception(f"Error running model for task {task_id}")
                            continue

                        # Create the prediction in Label Studio
                        logging.info(f"Creating prediction for task {task_id}")
                        logging.debug(prediction)
                        try:
                            resp = session.post(
                                predict_url,
                                json=(
                                    prediction
                                    | {"task": task_id, "project": project_id}
                                ),
                                timeout=10,
                            )
                            resp.raise_for_status()
                        except requests.exceptions.HTTPError:
                            logging.exception(
                                f"Error creating prediction for task {task_id}"
                            )
                    else:
                        logging.info(f"Skipping task {task_id}, existing prediction(s)")

                    # Update the progress bar
                    bar.next()
            except Exception as e:
                logging.exception("Unexpected exception")
                break


if __name__ == "__main__":
    # Load environment variables from a file, .env by default
    load_dotenv()
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        level=os.getenv("LOG_LEVEL"),
        filename="predict.log",
        encoding="utf-8",
    )

    base_url = os.environ["LABEL_STUDIO_URL"]
    project_id = os.environ["PROJECT_ID"]
    access_token = os.environ["LABEL_STUDIO_ACCESS_TOKEN"]
    view_id = os.getenv("VIEW_ID")

    create_predictions(base_url, access_token, project_id, view_id)
