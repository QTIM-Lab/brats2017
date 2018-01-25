import os

def default_config(config=None):

    """ Put default behaviors here """

    if config is None:
        config = dict()

    # Data will be compressed in hdf5 format at these filepaths.
    config["train_dir"] = ''
    config["validation_dir"] = ''
    config["test_dir"] = ''

    # Data will be saved to these hdf5 files.
    config["hdf5_train"] = 'train.hdf5'
    config["hdf5_validation"] = 'validation.hdf5'
    config["hdf5_test"] = 'test.hdf5'

    # Overwrite settings.
    config["overwrite_trainval_data"] = True
    config['overwrite_test_data'] = True
    config["overwrite_model"] = True
    config["overwrite_training"] = True
    config["overwrite_prediction"] = True

    # Image Information
    # config["image_shape"] = (240, 240, 155)
    # config["image_shape"] = (120, 120, 78)

    # Patch Information
    config['patch_shape'] = (16, 16, 4)
    config['training_patches_multiplier'] = 20
    config['validation_patches_multiplier'] = 5 

    # Modalities. Always make input_groundtruth as list.
    config["training_modality_dict"] = {'input_modalities': ['ADC_pp.nii.gz', 'MTT_pp.nii.gz', 'rCBF_pp.nii.gz', 'rCBV_pp.nii.gz', 'TMax_pp.nii.gz', 'TPP_pp.nii.gz'],
                                        'ground_truth': ['groundtruth-label_raw.nii.gz']}
    config["test_modality_dict"] = {'input_modalities': ['ADC_pp.nii.gz', 'MTT_pp.nii.gz', 'rCBF_pp.nii.gz', 'rCBV_pp.nii.gz', 'TMax_pp.nii.gz', 'TPP_pp.nii.gz'],
                                        'ground_truth': ['groundtruth-label_raw.nii.gz']}

    config["regression"] = False

    # Path to save model.
    config["model_file"] = "./model_files/model.h5"

    # Model parameters
    config["downsize_filters_factor"] = 1
    config["decay_learning_rate_every_x_epochs"] = 20
    config["initial_learning_rate"] = 0.00001
    config["learning_rate_drop"] = 0.9
    config["n_epochs"] = 100

    # Model training parameters
    config["batch_size"] = 200

    # Model testing parameters. More than needed, most likely.
    # config['predictions_folder'] = os.path.abspath('./predictions')
    config['predictions_folder'] = None
    config['predictions_name'] = 'infarct_prediction'
    config['predictions_input'] = 'input_modalities'
    config['predictions_groundtruth'] = 'ground_truth'
    config['predictions_replace_existing'] = False
    config['prediction_output_num'] = 1

    # Threshold Functions
    def background_patch(patch):
        return float((patch['input_modalities'] == 0).sum()) / patch['input_modalities'].size == 1
    def brain_patch(patch):
        return float((patch['input_modalities'] != 0).sum()) / patch['input_modalities'].size > .01 and float((patch['ground_truth'] == 1).sum()) / patch['ground_truth'].size < .01
    def roi_patch(patch):
        return float((patch['ground_truth'] == 1).sum()) / patch['ground_truth'].size > .01

    config["patch_extraction_conditions"] = [[background_patch, .02], [brain_patch, .199], [roi_patch, .8]]

    return config

def test_config(config=None):

    if config is None:
        config = default_config()

    config["overwrite_trainval_data"] = False
    config['overwrite_test_data'] = True
    config["overwrite_model"] = False
    config["overwrite_training"] = False
    config["overwrite_prediction"] = True

    return config

def predict_config(config=None):

    if config is None:
        config = default_config()

    config["overwrite_trainval_data"] = False
    config['overwrite_test_data'] = False
    config["overwrite_model"] = False
    config["overwrite_training"] = False
    config["overwrite_prediction"] = True

    return config

def train_config(config=None):

    if config is None:
        config = default_config()

    config["overwrite_trainval_data"] = False
    config['overwrite_test_data'] = False
    config["overwrite_model"] = True
    config["overwrite_training"] = True
    config["overwrite_prediction"] = False

    return config

def train_data_config(config=None):

    if config is None:
        config = default_config()

    config["overwrite_trainval_data"] = True
    config['overwrite_test_data'] = False
    config["overwrite_model"] = False
    config["overwrite_training"] = False
    config["overwrite_prediction"] = False

    return config

if __name__ == '__main__':
    pass

#config["train_test_split"] = .8
# config["input_modalities"] = ['FLAIR_p', 'T2_p', 'T1C_p', 'T1_p']
# config["input_groundtruth"] = ['GT_p']
# config['patches'] = True
# config['train_patch_num'] = 6000
# config['validation_patch_num'] = 3000
#def evaluate_model(assignments):
#    train(config)
# config["pool_size"] = (2, 2, 2)  # This determines what shape the images will be cropp_downsampleded/downsampled to.
# config["n_labels"] = 1  # not including background
# config["n_epochs"] = 50
# config["decay_learning_rate_every_x_epochs"] = 10
# config["initial_learning_rate"] = 0.00001
# config["learning_rate_drop"] = 0.5
# config["validation_split"] = 0.8
# config["smooth"] = 1.0
# # config["nb_channels"] = len(config["training_modalities"])
# # config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
# config["truth_channel"] = config["nb_channels"]
# config["background_channel"] = config["nb_channels"] + 1
# config["deconvolution"] = False  # use deconvolution instead of up-sampling. Requires keras-contrib.
# # divide the number of filters used by by a given factor. This will reduce memory consumption.
