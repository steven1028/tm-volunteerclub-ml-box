# Trend Micro Volunteer Club - AI for Kids Course
The repository provides the source code of AI for Kids course demostration.

## System Requirements
Following components are required to be installed:
- Python >= 3.7
- Keras >= 2.2.4
- keras-vggface >= 0.5
- opencv-python >= 4.0.0.21
- tensorflow >= 1.13.1

To install dependencies on your exist Python 3 environment, choose one of the following option:
1. Install on your global environment:
    ```
    # pip3 install sklearn tensorflow opencv-python matplotlib kears keras_vggface
    ```
2. Install on virtual environment:
    1. Navigate to the root folder of source repository, e.g.,
        ```
        # cd ~/tm-volunteerclub-ml-box
        ```
    2. Create a folder for virtual environment
        ```
        # mkdir .venv
        ```
    3. Construct virtual environment and activate
        ```
        # python3 -m venv ~/tm-volunteerclub-ml-box/.venv
        # source .venv/bin/activate
        ```
    4. Install the dependencies as usual:
        ```
        (.venv) # pip3 install sklearn tensorflow opencv-python matplotlib kears keras_vggface 
        ```
## Run the Sample Code
1. Navigate to the root folder of source repository, e.g.,
    ```
    # cd ~/tm-volunteerclub-ml-box
    ```
2. Execute the sample code as following
    ```
    python3 samples/FaceRecognitionDefaultVGG.py
    ```

## Todo
### Machine Learning Models
1. Face recognition, including:
    - Default VGG Face, cam be viewed as Celebrity similarity comparison
    - Customized VGG Face, can be viewed as self-trained face recognitor
2. Object detection, including:
    - Flower detection
    - Vehicle detection
    - ., etc.
3. Text classification / Clustering

### User-interface
- TBD
    - Web UI?
    - WxWidgets?
    - PyQt?