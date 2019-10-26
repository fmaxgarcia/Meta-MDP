
from ExperimentsCartpole import ExperimentsCartpole as EC
from ExperimentsAnimat import ExperimentsAnimat as EA
from optparse import OptionParser
import pickle


if __name__ == '__main__':
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-s", "--save", action="store", help="save directort", type="string")
    parser.add_option("-m", "--meta", action="store", help="pretrained meta policy", type="string", default=None)

    (options, args) = parser.parse_args()
    
    meta = options.meta
    save_dir = options.save
    
            
    assert(meta is not None)
    setups = ["./CustomEnvironments/maze5.txt", "./CustomEnvironments/maze6.txt", 
                "./CustomEnvironments/maze7.txt"]
    ##### Animat #######################
    EA.experiment_with_without_actions(options.meta, save_dir, setups, episodes=800, steps=1000 )

    #######################################
    