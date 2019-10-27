#A Meta-MDP Approach to Exploration for Lifelong Reinforcement Learning#

- To train the exploration policy use LearnMeta.py. For example for carpole:
	- python LearnMeta.py -e cartpole -d ./CartpoleData/cartpole_data.pkl -a 0.001 -b 0.010 -s ./MetaTrainingCartpole 
	(Note: for cartpole and animat we included pre-recorded data to simplify learning. It is not necessary to train the advisor)
 

- To get a baseline of a random exploration use RandomBaseline.py. Example:
	- python RandomBasline.py -e cartpole -d ./CartpoleData/cartpole_data.pkl -s ./RandomTrainingCartpole


Unfortunately, due to upload size we are not able to include the necessary files for the self-driving domain.
The changes to roboschool for creating model variations, require modifying the .xml model files after installation. 
