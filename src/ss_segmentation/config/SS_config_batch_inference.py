# The name of the folder which contains model files
MODEL_FOLDER = "drone_detection_test"

# Number of channels for input image (default:3)
NUM_CHANNELS = 3

# Number of classes used during training
NUM_CLASSES = 2

# Resolution of input image
IMG_HEIGHT = 512
IMG_WIDTH = 640

# Directory of model files
ARGS_LOAD_DIR = "/home/fla/ros_py35/src/image_test/" + MODEL_FOLDER
ARGS_LOAD_WEIGHTS = "/model_best.pth"
ARGS_LOAD_MODEL = "SS_network_design.py"

# Directory location to save inference visualizations
ARGS_SAVE_DIR = "/home/fla/ros_py35/src/image_test/"

# Directory which contains batch of input images
ARGS_INFERENCE_DIR = "/data/Docker_Data/drone_detection/data/processed/train/images/"

# Flag to save colorized masks
ARGS_SAVE_COLOR = 1

ARGS_NUM_WORKERS = 4
ARGS_BATCH_SIZE = 1
ARGS_CPU = False
