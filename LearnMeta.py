
from ExperimentsCartpole import ExperimentsCartpole as EC
from ExperimentsAnimat import ExperimentsAnimat as EA
from ExperimentsInvertedPendulum import ExperimentsInvertedPendulum as EIP
from ExperimentsHopper import ExperimentsHopper as EH 
from ExperimentsAnt import ExperimentsAnt as EAnt
from optparse import OptionParser
import pickle


if __name__ == '__main__':
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-e", "--environment", action="store", help="environment name", type="string", default="animat")
    parser.add_option("-d", "--data", action="store", help="initial data", type="string")
    parser.add_option("-s", "--save", action="store", help="initial data", type="string")
    parser.add_option("-a", "--alpha", action="store", help="meta alpha", type="float", default=1e-5)
    parser.add_option("-b", "--beta", action="store", help="meta alpha", type="float", default=1e-7)

    (options, args) = parser.parse_args()
    
    env = options.environment
    if options.data is not None:
        data = pickle.load(open(options.data, "rb"))
    alpha = options.alpha
    beta = options.beta
    save_dir = options.save

    if env.lower() == "cartpole":
    
        ##### CartPole #######################
        EC.RECORDED_DATA = data
        EC.experiment_train_meta(save_directory=save_dir, meta_alpha=alpha, meta_beta=beta)
        

        #######################################
    elif env.lower() == "animat":
            
        ##### Animat #######################
        EA.RECORDED_DATA = data
        EA.experiment_train_meta(save_directory=save_dir, meta_alpha=alpha, meta_beta=beta)

        #######################################

    elif env.lower() == "inverted_pendulum":

        models = ["inverted_pendulum_version1.xml", "inverted_pendulum_version2.xml", "inverted_pendulum_version3.xml"]
        EIP.experiment_train_meta(save_directory=save_dir, meta_alpha=alpha, meta_beta=beta, xml_models=models)

    elif env.lower() == "hopper":

        models = ["hopper.xml", "hopper_version1.xml", "hopper_version2.xml"]
        EH.experiment_train_meta(save_directory=save_dir, meta_alpha=alpha, meta_beta=beta, xml_models=models)

    elif env.lower() == "ant":

        models = ["ant.xml", "ant_version1.xml", "ant_version2.xml"]
        EAnt.experiment_train_meta(save_directory=save_dir, meta_alpha=alpha, meta_beta=beta, xml_models=models)
