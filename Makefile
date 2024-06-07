.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y text-to-speek || :
	@pip install -e .

run_create_inputs:
	python -c 'from app.utils import tokenize_pad_and_save_transcriptions; tokenize_pad_and_save_transcriptions()'
	python -c 'from app.utils import process_all_wavs_in_folder; process_all_wavs_in_folder()'
	python -c 'from app.utils import pad_and_save_durations; pad_and_save_durations()'
