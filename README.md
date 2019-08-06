# SepMe

Running Instructions:
1. `mlflow server` - this starts the mlflow server but it doesn't automatically create a conda env
2. `python -m SepMe.tasks.load_raw_data` works
3. `python -m SepMe.main` works but when it gets to running `load_raw_data` it breaks with the error `ModuleNotFoundError: No module named 'SepMe'`.

The only reason I run it as module is because I want `load_raw_data` to be able to import the logger from `SepMe/__init__.py` ?!