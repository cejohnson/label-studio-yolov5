# Label Studio YOLOv5

This repo is a translation layer between [YOLOv5](https://github.com/ultralytics/yolov5) and [Label Studio](https://github.com/HumanSignal/label-studio). It is not officially associated with either project in any way.

The translation layer can be used in a [Label Studio ML backend](https://labelstud.io/guide/ml), or it can be used in conjunction with the [Label Studio API](https://labelstud.io/api).

## Environment

Regardless of use case, set up your environment by copying `example.env` to `.env` and filling in the necessary information. This `.env` file will then be automatically used for different workflows.

## Label Studio Machine Learning Backend

1. Build the container for your architecture.
As of November 2023, most computers use the x86/64 architecture. A notable exception is Macs using Apple Silicon (M1/M2/M3/...).

    x86/64:  
    ```bash
    docker build -t label-studio-yolov5 .
    ```

    ARM64:
    ```bash
    docker build -f Dockerfile.arm -t label-studio-yolov5 .
    ```

2. Start Machine Learning backend on `http://localhost:9090`

    ```bash
    docker-compose up
    ```

3. Validate that backend is running.

    ```bash
    $ curl http://localhost:9090/health
    {"status":"UP"}
    ```

4. Connect to the backend from Label Studio: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.

## Running the Prediction Script

The predict script can be used to create predictions for Label Studio tasks using the Label Studio API. It will iterate through tasks for a given project (and optionally view/tab) and create predictions for each. Configuration is done through the `.env` file.

### Prerequisites

1. Clone the [YOLOv5 repo](https://github.com/ultralytics/yolov5) and update `.env` with its path.

2. Create and use a virtual environment (optional but *highly* recommended).

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

### Running

1. Install dependencies.

    ```bash
    pip install -r requirements.txt -r local_requirements.txt
    ```

2. Run the script.

    ```bash
    python predict.py
    ```

    Options:

    Dry run (doesn't create predictions in Label Studio but performs all other processing).
    ```bash
    python predict.py --dry-run
    ```

    Specify env file:
    ```bash
    python predict.py --env-file ENV_FILE
    ```