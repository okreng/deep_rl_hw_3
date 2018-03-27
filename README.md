## Important note for repository
DO NOT use the "git add ." command after modifying files
The ./env folder is ignored because it is enormous and cannot be uploaded
use "git add -u" instead
If new files are added in the workspace use git add <filename> to add them

## Setup:  
I was able to install the environment using the following commands:

sudo apt-get install swig (if not already using swig 3.0.8 - can find out w/ swig --version)
virtualenv env
source env/bin/activate
pip install -U -r requirements.txt

pip install box2d-py


