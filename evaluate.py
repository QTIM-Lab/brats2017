import os
import glob
import numpy as np
import csv

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy

def dice_spreadsheet(input_directories, ground_truth, comparison_files, output_csv='Edema_DICE'):

    output_numpy_list = []

    for directory_idx, directory in enumerate(input_directories):

        directory_list = glob.glob(os.path.join(directory, '*/'))

        output_numpy_list += [np.zeros((len(directory_list), len(comparison_files) +1), dtype=object)]

        output_numpy_list[directory_idx][0,0] = 'Patients'

        for subdir_idx, sub_directory in enumerate(directory_list):

            output_numpy_list[directory_idx][subdir_idx,0] = os.path.dirname(sub_directory)

            for vol_idx, volume in enumerate(comparison_files):



                input_volume_filename = os.path.join(sub_directory, volume)
                ground_truth_filename = os.path.join(sub_directory, ground_truth)

                try:
                    output_numpy_list[directory_idx][subdir_idx,1+vol_idx] = calculate_prediction_dice(input_volume_filename, ground_truth_filename)
                    print output_numpy_list[directory_idx][subdir_idx,:]
                except:
                    output_numpy_list[directory_idx][subdir_idx,1+vol_idx] = ''

    final_numpy = output_numpy_list[0]
    if len(output_numpy_list) > 1:
        for output in output_numpy_list[1:]:
            final_numpy = np.vstack((final_numpy, output))

    print final_numpy

    with open(output_csv, 'wb') as writefile:
        csvfile = csv.writer(writefile, delimiter=',')
        for row in final_numpy:
            csvfile.writerow(row)
                
    return

def calculate_prediction_dice(label_volume_1, label_volume_2):

    label_volume_1, label_volume_2, = convert_input_2_numpy(label_volume_1), convert_input_2_numpy(label_volume_2)

    im1 = np.asarray(label_volume_1).astype(np.bool)
    im2 = np.asarray(label_volume_2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

if __name__ == '__main__':

    input_directories = ['/mnt/jk489/sharedfolder/BRATS2017/Val']
    ground_truth = 'full_edemamask_pp.nii.gz'
    comparison_files = ['edema_upsampled_preloaded_3_1_prediction-label.nii.gz', 'full_edemamask_pp_downsampled_nn.nii.gz']
    output_csv = 'dice_comparison.csv'
    dice_spreadsheet(input_directories, ground_truth, comparison_files, output_csv)