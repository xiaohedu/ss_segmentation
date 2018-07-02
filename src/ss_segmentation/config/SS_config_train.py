# Currently input channels are default to 3
NUM_CHANNELS 		= 3

# Input image dimensions
IMG_HEIGHT		= 512
IMG_WIDTH 		= 640

ENC_SCALEDOWN		= 8

OPT_LEARNING_RATE_INIT 	= 5e-4

OPT_BETAS 		= (0.9, 0.999)
OPT_EPS_LOW 		= 1e-08
OPT_WEIGHT_DECAY 	= 1e-4

# For unbiased label rebalancing
NUM_CLASSES             = 2
CLASS_0_WEIGHT		= 0.003
CLASS_1_WEIGHT		= 0.997

ARGS_NUM_WORKERS	= 4
ARGS_BATCH_SIZE		= 5
ARGS_NUM_EPOCHS 	= 5
ARGS_IOU_TRAIN		= True
ARGS_IOU_VAL		= True
ARGS_STEPS_LOSS		= 20
ARGS_MODEL		= "SS_network_design"
ARGS_STATE		= ""
ARGS_DECODER		= ""
ARGS_PRETRAINED_ENCODER = ""
ARGS_CUDA		= True

# Make sure that the {train|val} folders contain "images" and "labels" subdirectories
# --- To change this data layout :: SS_data_definition.py
# SAVE_FOLDER: Folder which will contain learned model + any metadata files
SAVE_FOLDER             = "model_test_2"
ARGS_TRAIN_DIR		= "/data/Docker_Data/drone_detection/data/processed/train"
ARGS_VAL_DIR		= "/data/Docker_Data/drone_detection/data/processed/val"
ARGS_SAVE_DIR           = "/data/Docker_Inference/drone_detection/model_files/" + SAVE_FOLDER

ARGS_EPOCHS_SAVE	= 0
ARGS_RESUME		= False


