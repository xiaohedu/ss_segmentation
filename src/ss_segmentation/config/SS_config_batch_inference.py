MODEL_FOLDER = "drone_detection_test"

NUM_CHANNELS = 3
NUM_CLASSES = 2
IMG_HEIGHT = 512
IMG_WIDTH = 640

ARGS_LOAD_DIR = "/home/fla/ros_py35/src/image_test/" + MODEL_FOLDER
ARGS_LOAD_WEIGHTS = "/model_best.pth"
ARGS_LOAD_MODEL = "SS_network_design.py"
ARGS_SAVE_DIR = "/home/fla/ros_py35/src/image_test/"
ARGS_INFERENCE_DIR = "/data/Docker_Data/drone_detection/data/processed/train/images/"
ARGS_SAVE_COLOR = 1
ARGS_NUM_WORKERS = 4
ARGS_BATCH_SIZE = 1
ARGS_CPU = False
