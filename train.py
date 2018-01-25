import tables
import os
import math

from time import gmtime, strftime
from functools import partial
from shutil import rmtree

from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler
from keras.utils import plot_model
from keras.backend import clear_session

#import config_files.isles_config as isles_config
#import config_files.edema_config as edema_config
#import config_files.tumor1_config as tumor1_config
#import config_files.tumor2_config as tumor2_config
#import config_files.nonenhancing1_config as nonenhancing1_config
#import config_files.nonenhancing2_config as nonenhancing2_config
#import config_files.downsampled_edema_config as downsampled_edema_config
#import config_files.upsample_config as upsample_config
#import config_files.upsample_preloaded_config as upsample_preloaded_config
#import config_files.old_edema_config as old_edema_config
#import config_files.fms_config as fms_config

from model import n_net_3d, u_net_3d, split_u_net_3d, w_net_3d, load_old_model, vox_net, parellel_unet_3d
from load_data import DataCollection
from data_generator import get_data_generator, get_patch_data_generator
from data_utils import pickle_dump, pickle_load
from predict import model_predict_patches_hdf5
from augment import *
from file_util import split_folder

def learning_pipeline(overwrite=False, delete=False, config=None, parameters=None):

    # Modifications to add in regardless of config file
    ####################################################
    # append_prefix_to_config(config, ["hdf5_train", "hdf5_validation", "hdf5_test"], 'downsample_')
    # append_prefix_to_config(config, ["model_file"], strftime("%Y-%m-%d_%H:%M:%S", gmtime()))    

    # config['predictions_name'] = 'perpetual_patch_32'
    # config['predictions_replace_existing'] = True
    # config['overwrite_trainval_data'] = True
    
    # config['patch_shape'] = (24, 24, 4)
    # config['training_patches_multiplier'] = 50
    # config['validation_patches_multiplier'] = 10

    # config["downsize_filters_factor"] = 1
    # config["initial_learning_rate"] = 0.00001
    # config["regression"] = True
    # config["n_epochs"] = 200
    # config["batch_size"] = 1

    # config["image_shape"] = None

    fill_config_keys(config=config)
    update_config(config=config, parameters=parameters)
    create_directories(delete=delete, config=config)

    modality_dict = config['training_modality_dict']
    validation_files = []

    # Load training and validation data.
    if config['overwrite_trainval_data'] or not os.path.exists(os.path.abspath(config["hdf5_train"])):

        print 'WRITING DATA', '\n'

        flip_augmentation_group = AugmentationGroup({'input_modalities': Flip_Rotate_2D(flip=True, rotate=False), 'ground_truth': Flip_Rotate_2D(flip=True, rotate=False)}, multiplier=2)

        # Find Data
        training_data_collection = DataCollection(config['train_dir'], modality_dict, brainmask_dir=config['brain_mask_dir'], roimask_dir=config['roi_mask_dir'], patch_shape=config['patch_shape'])
        training_data_collection.fill_data_groups()

        # # Training - with patch augmentation
        if not config['perpetual_patches']:
            training_patch_augmentation = ExtractPatches(config['patch_shape'], config['patch_extraction_conditions'])
            training_patch_augmentation_group = AugmentationGroup({'input_modalities': training_patch_augmentation, 'ground_truth': training_patch_augmentation}, multiplier=config['training_patches_multiplier'])
            training_data_collection.append_augmentation(training_patch_augmentation_group)

        training_data_collection.append_augmentation(flip_augmentation_group)
        training_data_collection.write_data_to_file(output_filepath = config['hdf5_train'], save_masks=config["overwrite_masks"])

        if config['validation_dir'] is not None:

            # Validation - with patch augmentation
            validation_data_collection = DataCollection(config['validation_dir'], modality_dict, brainmask_dir=config['brain_mask_dir'], roimask_dir=config['roi_mask_dir'], patch_shape=config['patch_shape'])
            validation_data_collection.fill_data_groups()

            if not config['perpetual_patches']:
                validation_patch_augmentation = ExtractPatches(config['patch_shape'], config['patch_extraction_conditions'])
                validation_patch_augmentation_group = AugmentationGroup({'input_modalities': validation_patch_augmentation, 'ground_truth': validation_patch_augmentation}, multiplier=config['validation_patches_multiplier'])
                validation_data_collection.append_augmentation(validation_patch_augmentation_group)
            
            validation_data_collection.append_augmentation(flip_augmentation_group)
            validation_data_collection.write_data_to_file(output_filepath = config['hdf5_validation'], save_masks=config["overwrite_masks"], store_masks=True)

    # Create a new model if necessary. Preferably, load an existing one.
    if not config["overwrite_model"] and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        model = u_net_3d(input_shape=(len(modality_dict['input_modalities']),) + config['patch_shape'], output_shape=(len(modality_dict['ground_truth']),) + config['patch_shape'], downsize_filters_factor=config['downsize_filters_factor'], initial_learning_rate=config['initial_learning_rate'], regression=config['regression'], num_outputs=(len(modality_dict['ground_truth'])))

    plot_model(model, to_file='model_image.png', show_shapes=True)

    # Create data generators and train the model.
    if config["overwrite_training"]:
        
        # Get training and validation generators, either split randomly from the training data or from separate hdf5 files.
        if config["validation_dir"] is None:
            open_validation_hdf5 = []
            validation_generator, num_validation_steps = None, None
        elif os.path.exists(os.path.abspath(config["hdf5_validation"])):
            print "Validation data found"
            open_validation_hdf5 = tables.open_file(config["hdf5_validation"], "r")
            if config['perpetual_patches']:
                validation_generator, num_validation_steps = get_patch_data_generator(open_validation_hdf5, batch_size=config["validation_batch_size"], data_labels = ['input_modalities', 'ground_truth'], patch_multiplier=config['validation_patch_multiplier'], patch_shape=config['patch_shape'], roi_ratio=config['roi_ratio'])
            else:
                validation_generator, num_validation_steps = get_data_generator(open_validation_hdf5, batch_size=config["validation_batch_size"], data_labels = ['input_modalities', 'ground_truth'])
                print validation_generator
                print num_validation_steps
        else:
            print "Validation data not found! Exiting pipeline."

        open_train_hdf5 = tables.open_file(config["hdf5_train"], "r")

        if config['perpetual_patches']:
            train_generator, num_train_steps = get_patch_data_generator(open_train_hdf5, batch_size=config["training_batch_size"], data_labels = ['input_modalities', 'ground_truth'], patch_multiplier=config['training_patch_multiplier'], patch_shape=config['patch_shape'], roi_ratio=config['roi_ratio'], num_epoch_steps=config['num_epoch_steps'], num_subepochs=config['num_subepochs'], num_subepoch_patients=config['num_subepoch_patients'])
        else:
            train_generator, num_train_steps = get_data_generator(open_train_hdf5, batch_size=config["training_batch_size"], data_labels = ['input_modalities', 'ground_truth'])

        # Train model.. TODO account for no validation
        train_model(model=model, model_file=config["model_file"], training_generator=train_generator, validation_generator=validation_generator, steps_per_epoch=num_train_steps, validation_steps=num_validation_steps, initial_learning_rate=config["initial_learning_rate"], learning_rate_drop=config["learning_rate_drop"], learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])

        # Close training and validation files, no longer needed.
        open_train_hdf5.close()
        if validation_files:
            open_validation_hdf5.close()

    # Load testing data
    if config['overwrite_test_data'] or not os.path.exists(os.path.abspath(config["hdf5_test"])):
        
        modality_dict = config['test_modality_dict']

        testing_data_collection = DataCollection(config['test_dir'], modality_dict)
        testing_data_collection.fill_data_groups()

        testing_data_collection.write_data_to_file(output_filepath = config['hdf5_test'], save_masks=False, store_masks=False)

    # Run prediction step.
    if config['overwrite_prediction']:
        open_test_hdf5 = tables.open_file(config["hdf5_test"], "r")
        model_predict_patches_hdf5(data_file=open_test_hdf5, input_data_label=config['predictions_input'], patch_shape=config['patch_shape'], output_directory=config['predictions_folder'], output_name=config['predictions_name'], ground_truth_data_label=config['predictions_groundtruth'], model=model, replace_existing=config['predictions_replace_existing'])

    clear_session()

def create_directories(delete=False, config=None):

    # Create required directories
    for directory in [config['model_file'], config['hdf5_train'], config['hdf5_test'], config['hdf5_validation'], config['predictions_folder'], config['brain_mask_dir'], config['roi_mask_dir']]:
        if directory is not None:
            directory = os.path.abspath(directory)
            if not os.path.isdir(directory):
                directory = os.path.dirname(directory)
            if delete:
                rmtree(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)

def update_config(config, parameters):

    if parameters is None:
        return

    for key in parameters.keys():
        config[key] = parameters[key]

def fill_config_keys(config):

    if 'num_subepochs' not in config.keys():
        config['num_subepochs'] = None

    if 'num_epoch_steps' not in config.keys():
        config['num_epoch_steps'] = 10


def append_prefix_to_config(config, keys, prefix):

    for key in keys:
        config[key] = '/'.join(str.split(config[key], '/')[0:-1]) + '/' + prefix + str.split(config[key], '/')[-1]

def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps, initial_learning_rate, learning_rate_drop, learning_rate_epochs, n_epochs):

    if validation_generator is None:
        model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs, pickle_safe=True, callbacks=get_callbacks(model_file, initial_learning_rate=initial_learning_rate, learning_rate_drop=learning_rate_drop,learning_rate_epochs=learning_rate_epochs))
    else:
        model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs, validation_data=validation_generator, validation_steps=validation_steps, pickle_safe=True, callbacks=get_callbacks(model_file, initial_learning_rate=initial_learning_rate, learning_rate_drop=learning_rate_drop,learning_rate_epochs=learning_rate_epochs))

    model.save(model_file)

""" The following three functions/classes are mysterious to me.
"""

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

class SaveLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")

def get_callbacks(model_file, initial_learning_rate, learning_rate_drop, learning_rate_epochs, logging_dir="."):

    """ Currently do not understand callbacks.
    """

    model_checkpoint = ModelCheckpoint(model_file, monitor="loss", save_best_only=True)
    logger = CSVLogger(os.path.join(logging_dir, "training.log"))
    history = SaveLossHistory()
    scheduler = LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate, drop=learning_rate_drop, epochs_drop=learning_rate_epochs))
    return [model_checkpoint, logger, history, scheduler]

def pipeline(config_name='', mode='default'):

    from configs import config_map
    config_type = config_map[config_name]
    
    if config_type:
        
        if mode == 'default':       
            config_mode = config_type.default_config()
        elif mode == 'train':
            config_mode = config_type.train_config()
        elif mode == 'predict':
            config_mode = config_type.predict_config()
        elif mode == 'test':
            config_mode = config_type.test_config()
        elif mode == 'train_data':
            config_mode = config_type.train_data_config()            
        else:
            raise NotImplementedError("Config option '{}' not implemented!".format(mode))
        
        learning_pipeline(config=config_mode, overwrite=False)

    else:
        pass
        # ISLES CONFIGURATION OPTIONS

        # learning_pipeline(config=isles_config.default_config(), overwrite=False)
        # learning_pipeline(config=isles_config.train_config(), overwrite=False)
        # learning_pipeline(config=isles_config.predict_config(), overwrite=False)


        # BRATS CONFIGURATION OPTIONS

        # learning_pipeline(config=old_edema_config.predict_config(), overwrite=False)

        # learning_pipeline(config=edema_config.default_config(), overwrite=False)
        # learning_pipeline(config=edema_config.train_config(), overwrite=False)
        # learning_pipeline(config=edema_config.predict_config(), overwrite=False)

        # learning_pipeline(config=downsampled_edema_config.default_config(), overwrite=False)

        # learning_pipeline(config=fms_config.test_config(), overwrite=False)
        
        # learning_pipeline(config=upsample_config.test_config(), overwrite=False)  
        # learning_pipeline(config=upsample_preloaded_config.test_config(), overwrite=False)

        # learning_pipeline(config=tumor1_config.train_config(), overwrite=False)
        # learning_pipeline(config=nonenhancing1_config.default_config(), overwrite=False)
        # learning_pipeline(config=nonenhancing1_config.test_config(), overwrite=False)
        # learning_pipeline(config=tumor2_config.train_data_config(), overwrite=False)
        # learning_pipeline(config=tumor2_config.test_config(), overwrite=False)
        # learning_pipeline(config=nonenhancing2_config.train_data_config(), overwrite=False)
        # learning_pipeline(config=nonenhancing2_config.test_config(), overwrite=False)
        # learning_pipeline(config=upsample_config.train_data_config(), overwrite=False)

        # learning_pipeline(config=edema_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})
        # learning_pipeline(config=tumor1_config.test_config(), overwrite=False, parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})
        # learning_pipeline(config=nonenhancing1_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})    
        # learning_pipeline(config=tumor2_config.test_config(), overwrite=False, parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})
        # learning_pipeline(config=nonenhancing2_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})

        # learning_pipeline(config=edema_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})
        # learning_pipeline(config=tumor1_config.test_config(), overwrite=False, parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})
        # learning_pipeline(config=nonenhancing1_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})    
        # learning_pipeline(config=tumor2_config.test_config(), overwrite=False, parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})
        # learning_pipeline(config=nonenhancing2_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})

        # split_folder('/mnt/jk489/sharedfolder/BRATS2017/Val', .2, ['/mnt/jk489/sharedfolder/BRATS2017/Val_Train', '/mnt/jk489/sharedfolder/BRATS2017/Val_Val'])
        # learning_pipeline(config=reconciliation_config.train_config(), overwrite=False)
        # learning_pipeline(config=reconciliation_config.test_config(), overwrite=False)


if __name__ == '__main__':

    pass