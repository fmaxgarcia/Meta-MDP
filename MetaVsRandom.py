
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
    parser.add_option("-s", "--save", action="store", help="save directort", type="string")
    parser.add_option("-a", "--actor", action="store", help="pretrained meta policy", type="string", default=None)
    parser.add_option("-c", "--critic", action="store", help="pretrained meta policy", type="string", default=None)

    (options, args) = parser.parse_args()
    
    env = options.environment
    save_dir = options.save

    print(env)
    if env.lower() == "cartpole":
    
        ##### CartPole #######################
        # assert(meta is not None)
        setups = [{"force" : 12.0, "pole_length" : 0.7, "masscart" : 0.2, "masspole" : 0.4},            
                    {"force" : 5.0, "pole_length" : 0.25, "masscart" : 2.0, "masspole" : 0.2},
                    {"force" : 2.0, "pole_length" : 0.2, "masscart" : 5.0, "masspole" : 1.0},
                    ]

        EC.experiment_meta_vs_random(options.actor, options.critic, save_dir, setups, episodes=501, steps=1000 )
        

        #######################################
    elif env.lower() == "animat":
            
        # assert(meta is not None)
        setups = ["./CustomEnvironments/maze5.txt", "./CustomEnvironments/maze6.txt", 
                  "./CustomEnvironments/maze7.txt"]
        ##### Animat #######################
        # EA.experiment_meta_vs_random(options.meta, save_dir, setups, episodes=250, steps=600 )
        EA.experiment_meta_vs_random(options.actor, options.critic, save_dir, setups, episodes=250, steps=800 )

        #######################################
    elif env.lower() == "inverted_pendulum":

        models = ["inverted_pendulum_version4.xml", "inverted_pendulum_version5.xml", "inverted_pendulum_version6.xml"]
        EIP.experiment_meta_vs_random(options.actor, options.critic, save_dir, xml_models=models, episodes=250, steps=500 )

    elif env.lower() == "hopper":

        models = ["hopper_version3.xml", "hopper_version4.xml"]
        EH.experiment_meta_vs_random(options.actor, options.critic, save_dir, xml_models=models, episodes=500, steps=500 )

    elif env.lower() == "ant":

        models = ["ant_version3.xml", "ant_version4.xml"]
        EAnt.experiment_meta_vs_random(options.actor, options.critic, save_dir, xml_models=models, episodes=500, steps=500 )
    