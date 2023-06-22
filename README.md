## Personal ANI Training Engine

To install:

`git clone git@github.com:khuddzu/personal_trainer.git`

`pip install -e .` 

### Initiate protocol

To initiate the personal trainer protocol 

`personal_trainer setup`

### Format

training\_center: main directory
	- logs : the directory that all of your trained models will be stored in. Currently the naming format is datetime\_projectname. 
	- models: the directory that will contain all of your model codes. Currently the model which you want to use must be located in a file called nets.py. Simply change the source if you want this to be different.
	- editor.ini: template configuration file for your networks.
	- template\_trainer.py: template code that calls variables from your configuration file and feeds them to protocol trainer, to train your network.
	- template\_model\_loader.py: template code that loads your trained model, for use.
	- script.sh: template script to send your training job to the queue.
