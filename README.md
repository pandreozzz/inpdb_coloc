## Develop stuff
**IMPORTANT: Create your own code development branch and make pull requests to main**

## Notes
In the folder `./notes` we could put shared info like meeting summaries, tasks etc. Pushing to main is probably OK here

## Usage example
Check out the jupyter notebook introduction.ipynb
to check the main API usage.

## virtualenv and requirements
Create a virtualenv
```
% python3 -m venv .local_venv
% source .local_venv/bin/activate
```
install dependencies
```
% pip install -r requirements.txt
```

## Data population
Checkout the notebook `introduction.ipynb`. The idea is that there is a folder containing most data `_STORAGE`, that you just download and place somewhere with enough free space. Or you can just reference the storage folder if it is somewhere in the filesystem, even if you only have read access. Then you have an `_ACTIVE` folder where the actual synthesis or interpolated data are located, these are typically generated the first time you run the code to make future runs much quicker. The `_ACTIVE` folder can be anywhere, but you would need some GB of free space. Finally, the `data` folder (by default in the project directory) is where the symlinks to the `_ACTIVE` subdirectories are created, so no actual data is stored there. To populate data, you use the `create_data.sh` script:
```bash
$ ./create_data.sh --help
$ ./create_data.sh --active-path <path_to_active_folder> --storage-path <path_to_storage_folder>
```
this considers by default data to be `./data` and will clean up both the data and `_ACTIVE` folders beforehand.
