import glob
import os

from shutil import move, copy, rmtree, copytree
from random import shuffle

# from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
# from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

# def preview_full_label(directory='/mnt/jk489/sharedfolder/BRATS2017/Validation', labels=['edema_prediction.nii.gz','tumor_prediction_from_edema.nii.gz','nonenhancing_prediction_from_edema_tumor.nii.gz'], output_filename='fine_tuned_prediction-label'):
    
#     dir_list = glob.glob(os.path.join(directory, '*/'))

#     for patient_dir in dir_list:

#         if not os.path.exists(os.path.join(patient_dir, output_filename)):

#             print patient_dir
#             edema = convert_input_2_numpy(os.path.join(patient_dir, labels[0]))
#             tumor = convert_input_2_numpy(os.path.join(patient_dir, labels[1]))
#             nonenhancing = convert_input_2_numpy(os.path.join(patient_dir, labels[2]))

#             edema[tumor == 1] = 2
#             edema[nonenhancing == 1] = 3

#             save_numpy_2_nifti(edema, os.path.join(patient_dir, 'edema_prediction.nii.gz'), os.path.join(patient_dir, output_filename))

# def rename_preview_label(directory='/mnt/jk489/sharedfolder/BRATS2017/Validation'):
    
#     dir_list = glob.glob(os.path.join(directory, '*/'))

#     for patient_dir in dir_list:

#         move(os.path.join(patient_dir, 'andrew_convenience_combination.nii.gz'), os.path.join(patient_dir, 'andrew_convenience_combination-label.nii.gz'))

def nifti_splitext(input_filepath):

    """ os.path.splitext splits a filename into the part before the LAST
        period and the part after the LAST period. This will screw one up
        if working with, say, .nii.gz files, which should be split at the
        FIRST period. This function performs an alternate version of splitext
        which does just that.

        TODO: Make work if someone includes a period in a folder name (ugh).

        Parameters
        ----------
        input_filepath: str
            The filepath to split.

        Returns
        -------
        split_filepath: list of str
            A two-item list, split at the first period in the filepath.

    """

    split_filepath = str.split(input_filepath, '.')

    if len(split_filepath) <= 1:
        return split_filepath
    else:
        return [split_filepath[0], '.' + '.'.join(split_filepath[1:])]

def replace_suffix(input_filepath, input_suffix, output_suffix, suffix_delimiter=None):

    """ Replaces an input_suffix in a filename with an output_suffix. Can be used
        to generate or remove suffixes by leaving one or the other option blank.

        TODO: Make suffixes accept regexes. Can likely replace suffix_delimiter after this.
        TODO: Decide whether suffixes should extend across multiple directory levels.

        Parameters
        ----------
        input_filepath: str
            The filename to be transformed.
        input_suffix: str
            The suffix to be replaced
        output_suffix: str
            The suffix to replace with.
        suffix_delimiter: str
            Optional, overrides input_suffix. Replaces whatever 
            comes after suffix_delimiter with output_suffix.

        Returns
        -------
        output_filepath: str
            The transformed filename
    """

    split_filename = nifti_splitext(input_filepath)

    if suffix_delimiter is not None:
        input_suffix = str.split(split_filename[0], suffix_delimiter)[-1]

    if input_suffix not in os.path.basename(input_filepath):
        print 'ERROR!', input_suffix, 'not in input_filepath.'
        return []

    else:
        if input_suffix == '':
            prefix = split_filename[0]
        else:
            prefix = input_suffix.join(str.split(split_filename[0], input_suffix)[0:-1])
        prefix = prefix + output_suffix
        output_filepath = prefix + split_filename[1]
        return output_filepath

def split_folder(directory, split, output_folders, copy=True, delete=True):

    for output_folder in output_folders:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        elif delete:
            rmtree(output_folder)

    folder_list = glob.glob(os.path.join(directory, '*/'))

    stopping_point = int(split*len(folder_list))

    for folder_idx, folder in enumerate(folder_list):

        if folder_idx > stopping_point:
            temp_output = output_folders[0]
        else:
            temp_output = output_folders[1]

        print os.path.join(temp_output, str.split(folder, '/')[-2])

        if copy:
            copytree(folder, os.path.join(temp_output, str.split(folder, '/')[-2]))
        else:
            move(folder, os.path.join(temp_output, str.split(folder, '/')[-2]))

if __name__ == '__main__':
    # preview_full_label(directory='/mnt/jk489/sharedfolder/BRATS2017/Validation', labels=['finetuned_prediction_0.nii.gz','finetuned_prediction_1.nii.gz','finetuned_prediction_2.nii.gz'], output_filename='fine_tuned_prediction-label.nii.gz')
    # rename_preview_label()
    split_folder('/mnt/jk489/sharedfolder/BRATS2017/Val', .2, ['/mnt/jk489/sharedfolder/BRATS2017/Val_Train', '/mnt/jk489/sharedfolder/BRATS2017/Val_Val'])