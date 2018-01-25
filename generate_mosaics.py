import matplotlib
matplotlib.use('Agg')
from qtim_tools.qtim_visualization.image import create_mosaic
import sys
from os import makedirs
from os.path import join, basename, isdir, isfile
from glob import glob
from functools import partial
import multiprocessing

def mosaic_one(case_dir, args):

	out_dir = args['out_dir']

	for i, (mri_name, label_name) in enumerate(vol_labels):

		if i == 0:
			continue

		mri_volume = join(case_dir, mri_name)
		label_volume = join(case_dir, label_name)

		out_case_dir = join(out_dir, basename(case_dir))
		if not isdir(out_case_dir):
			makedirs(out_case_dir)

		outfile = join(out_case_dir, 'label_{}.png'.format(i))
		# if not isfile(outfile):
		try:
			create_mosaic(mri_volume, outfile=outfile, label_volume=label_volume, step=1)
			print outfile
		except Exception:
			print "Unable to make mosaic for '{}'".format(outfile)

test_path = sys.argv[1]
test_out_dir = sys.argv[2]

vol_labels = [('FLAIR_pp.nii.gz', 'upsample_edema_2_prediction-label.nii.gz'),
			  ('T1post_pp.nii.gz', 'finetune_enh_final_prediction-label.nii.gz'),
			  ('T2_pp.nii.gz', 'finetune_nonenh_final_prediction-label.nii.gz')]

test_cases = glob(join(test_path, '*'))
pool = multiprocessing.Pool(processes=8)
subprocess = partial(mosaic_one, args={'out_dir': test_out_dir})
_ = pool.map(subprocess, test_cases)
