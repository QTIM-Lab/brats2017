import sys
from os.path import join, isdir, isfile, basename
from glob import glob
import nibabel
import numpy as np
from image_utils import nifti_2_numpy, save_numpy_2_nifti, predictions_to_labelmap


sample_data = sys.argv[1]
all_samples = glob(join(sample_data, '*'))

thresholds = [.5, .5, .5]

input_prediction_names = ['upsample_edema_2_prediction-label.nii.gz',
						  'enhancing_final_prediction-label.nii.gz',
						  'nonenhancing_final_prediction-label.nii.gz']

ground_truth_name = 'seg_pp.nii.gz'
label_mappings = [(2, 1), (4, 2), (1, 3)]  # edema: 1, enhancing: 2, non-enhancing: 3


for sample_dir in all_samples:

	if not isdir(sample_dir):
		continue

	finetune_input = join(sample_dir, 'finetune_input-labels.nii.gz')
	print "Generating fine-tune inputs for {}".format(basename(sample_dir))
	sample_preds = [join(sample_dir, x) for x in input_prediction_names]
	labelmap = predictions_to_labelmap(sample_preds, thresholds).astype(np.uint8)

	print finetune_input
	save_numpy_2_nifti(labelmap, reference_nifti_filepath=sample_preds[0], output_filepath=finetune_input)

	finetune_output = join(sample_dir, 'finetune_groundtruth.nii.gz')
	print "Generating fine-tune ground truth for {}".format(basename(sample_dir))
	seg_volume, seg_affine = nifti_2_numpy(join(sample_dir, ground_truth_name), return_affine=True)

	groundtruth_volume = np.zeros(shape=seg_volume.shape)
	for old_label, new_label in label_mappings:
		groundtruth_volume[seg_volume == old_label] = new_label

	print finetune_output
	save_numpy_2_nifti(groundtruth_volume.astype(np.uint8), reference_affine=seg_affine, output_filepath=finetune_output)

