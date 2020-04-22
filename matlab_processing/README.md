This folder has three MAIN files
1) "align_batch", which performs alignment of a set of MS images.
2) "mcml_cardiac", which performs the MCML/SVM classification.
3) "preprocess", which performs the tissue and peak selection prior to training.

1) "align_batch" calls "remapping", which uses "MZMap", which then uses "binarymapiter".
2) "mcml_cardiac" is mostly self-contained, so it doesn't call anything else.
3) "preprocess" is also self-contained and only calls ionpick to do the peak selection by template ion.

"visualize_sample" was used to generate an image of the sample which shows which pixels were selected for training.

***generate_batch traverses the workspace for '.mat' files, imports them, adds them to a batch and performs alignment.
It handles differently sized mcvecs by zero-padding smaller ones to fit the larger ones, however, it isn't robustly
coded right now, so that part should be altered if it becomes necessary to handle data with many different mcvec
sizes
