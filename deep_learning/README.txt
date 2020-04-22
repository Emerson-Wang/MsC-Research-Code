The key file is "cnn_run.py". This performs the CNN stage of the stacked learning system, producing a CNN model for
each sample which can produce a CNN score. This function has a data import function, a testing function, and calls
a training function which is in a separate file called "training_GPU.py", which I left as a standalone function because
of the effort of re-factoring it to fit the "cnn_run" script. "training_GPU" uses the files generated in the import
function to train a model, which it then saves. This model is then imported during the testing function to produce
the final classifications.

The folder is split into several data types, the majority of which are .txt files

1) .h5 files are saved CNN models
2) .txt files are almost all data files, either for input data or for data which has been split into training/testing
sets or for records of CNN results. 
	"trials_results" is the most important of the CNN results, as this is the final testing results I used as the
	CNN scores for the rest of stacked learning. I've moved most of the rest of the experiment results to a separate
	folder, but without context to a lot of them they aer probably not useful.
3) .py the code

TRAINING FOR POAF IN CNN
I also had to test using the CNN to classify POAF using the demographics and NH scores to show that deep learning
failed on the problem. My process for this was quick and dirty and therefore difficult to convey in a useful manner.
The key file for this is "demog_CNN.py". I attempted to make it work at least somewhat automated, so it should work
fine if you just run it. However, it may have some problems related to my specific workflow. If this is the case, let me
know and I can sort it out. I believe it works similarly to "cnn_run", except with an extra step where it runs a file
called "train_discriminator" after training the NH model.