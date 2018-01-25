import os
import numpy as np

from image_utils import save_numpy_2_nifti
from scipy.ndimage.interpolation import zoom
from shutil import rmtree

def dummy_data_generator(input_modalities=4, modality_dims=(8,8,8), dummy_data_folder = './dummy_data', train_test = [600, 10], noise=True):

    dummy_data_folder = os.path.abspath(dummy_data_folder)

    if os.path.exists(dummy_data_folder):
        rmtree(dummy_data_folder)
    os.mkdir(dummy_data_folder)

    folder_labels = [[train_test[0], 'train'], [train_test[1], 'test']]

    for cases, folder_label in folder_labels:

        dummy_data_subfolder = os.path.join(dummy_data_folder, folder_label)
        if not os.path.exists(dummy_data_subfolder):
            os.mkdir(dummy_data_subfolder)

        for case_num in xrange(cases):

            output_folder = os.path.join(dummy_data_subfolder, 'CASE_' + str(case_num).zfill(3))

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            case = []
            input_modality_list = []
            output_groundtruth = np.zeros(modality_dims)

            for mod_num in xrange(input_modalities):

                modality = np.random.random((2,2,2))
                modality = zoom(modality, [4,4,4], order=1)

                modality2 = np.zeros((2,2,2)) + .5
                modality2 = zoom(modality2, [4,4,4], order=1)

                # print np.sum((modality-modality2)**2)

                input_modality_list += [modality]
                # output_groundtruth = output_groundtruth + modality

            output_groundtruth = input_modality_list[0] - input_modality_list[1]**2 + input_modality_list[2] * input_modality_list[3]

            for mod_num, modality in enumerate(input_modality_list):
                if noise:
                    modality = modality + np.random.random(modality_dims)/5
                save_numpy_2_nifti(modality, '', os.path.join(output_folder, 'modality_' + str(mod_num) + '.nii.gz'))

            save_numpy_2_nifti(output_groundtruth, '', os.path.join(output_folder, 'groundtruth.nii.gz'))

def dummy_patch_data_generator(input_modalities=4, modality_dims=(100,100,100), dummy_data_folder = './dummy_data', train_test = [15, 5], noise=True):

    dummy_data_folder = os.path.abspath(dummy_data_folder)

    if os.path.exists(dummy_data_folder):
        rmtree(dummy_data_folder)
    os.mkdir(dummy_data_folder)

    folder_labels = [[train_test[0], 'train'], [train_test[1], 'test']]

    for cases, folder_label in folder_labels:

        dummy_data_subfolder = os.path.join(dummy_data_folder, folder_label)
        if not os.path.exists(dummy_data_subfolder):
            os.mkdir(dummy_data_subfolder)

        for case_num in xrange(cases):

            output_folder = os.path.join(dummy_data_subfolder, 'CASE_' + str(case_num).zfill(3))

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            case = []
            input_modality_list = []
            output_groundtruth = np.zeros(modality_dims)

            for mod_num in xrange(input_modalities):

                modality = np.zeros(modality_dims)
                modality_core = np.random.random((8,8,8))
                modality_core = zoom(modality_core, [8,8,8], order=1)
                modality[18:82, 18:82, 18:82] = modality_core

                # modality2 = np.zeros((2,2,2)) + .5
                # modality2 = zoom(modality2, [4,4,4], order=1)

                # print np.sum((modality-modality2)**2)

                input_modality_list += [modality]
                # output_groundtruth = output_groundtruth + modality

            output_groundtruth = input_modality_list[0] - input_modality_list[1]**2 + input_modality_list[2] * np.sin(input_modality_list[3])
            save_numpy_2_nifti(output_groundtruth, '', os.path.join(output_folder, 'groundtruth.nii.gz'))

            for mod_num, modality in enumerate(input_modality_list):
                if noise:
                    modality[18:82, 18:82, 18:82] = modality[18:82, 18:82, 18:82] + np.random.random((64,64,64))/5
                save_numpy_2_nifti(modality, '', os.path.join(output_folder, 'modality_' + str(mod_num) + '.nii.gz'))

if __name__ == '__main__':
    # dummy_data_generator()
    dummy_patch_data_generator()