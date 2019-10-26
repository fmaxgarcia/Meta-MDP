
from ExperimentsCartpole import ExperimentsCartpole as EC
from ExperimentsLander import ExperimentsLander as EL
from ExperimentsAnimat import ExperimentsAnimat as EA
from optparse import OptionParser
import pickle


if __name__ == '__main__':
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-e", "--environment", action="store", help="environment name", type="string", default="animat")
    parser.add_option("-d", "--data", action="store", help="initial data", type="string")
    parser.add_option("-s", "--save", action="store", help="initial data", type="string")
 
    (options, args) = parser.parse_args()
    
    env = options.environment
    data = pickle.load(open(options.data, "rb"))
    save_dir = options.save

    if env.lower() == "cartpole":
    
        ##### CartPole #######################
        EC.RECORDED_DATA = data
        EC.experiment_random_baseline(save_directory=save_dir)
        

        #######################################
    elif env.lower() == "animat":
            
        ##### Animat #######################
        EA.RECORDED_DATA = data
        EA.experiment_random_baseline(save_directory=save_dir)

        #######################################
    