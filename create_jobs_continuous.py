
alphas = [1e-3, 1e-4, 1e-5]
betas = [1e-2, 1e-3, 1e-4, 1e-5]

save = ["./MamlTrainingInvPendulum", "./MamlTrainingHopper", "./MamlTrainingAnt", 
        "./MetaTrainingPendulum", "./MetaTrainingHopper", "./MetaTrainingAnt"]
env = ["inverted_pendulum", "hopper", "ant"]

for k in range(6):
    s = save[k]
    e = env[k%3]
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            if k < 3:
                f = open("run_maml_%s_%d_%d.sh" %(e, i,j), "w") #Create shell to run the python program
            else:
                f = open("run_meta_%s_%d_%d.sh" %(e, i,j), "w") #Create shell to run the python program
            params = "-e %s -a %f -b %f -s %s_%d_%d" %(e, a, b, s, i, j)
            if k < 3:
                f.write("python LearnMaml.py %s" %(params)) #Command to run the python program
            else:
                f.write("python LearnMeta.py %s" %(params)) #Command to run the python program
            f.close()
            if k < 3:
                f = open("slurm_maml_%s_%d_%d.sh" %(e, i,j), "w") #Create shell with options for swam
            else:
                f = open("slurm_meta_%s_%d_%d.sh" %(e, i,j), "w") #Create shell with options for swam
            f.write("#! /bin/bash")
            f.write("\n#SBATCH --job-name=slurm_%s_%d_%d" %(e, i,j)) #Name for swarm job
            if k < 3:
                f.write("\n#SBATCH --output=result-maml_%s_%d_%d.out" %(e, i,j)) #Name of file where output is written
                f.write("\n#SBATCH --error=error-maml_%s_%d_%d.out" %(e, i,j)) #Name of file to log errors
            else:
                f.write("\n#SBATCH --output=result-meta_%s_%d_%d.out" %(e, i,j)) #Name of file where output is written
                f.write("\n#SBATCH --error=error-meta_%s_%d_%d.out" %(e, i,j)) #Name of file to log errors
            f.write("\n#SBATCH --nodes=1") #Number of nodes requested
            f.write("\n#SBATCH --cpus-per-task=8") #Number of cores per node requested
            f.write("\n#SBATCH --mem=5000") #Amount of memory requested in MB
            f.write("\n#SBATCH --partition=titanx-long") #Type of process to run (longq gives you a week)
            f.write("\n\n")
            if k < 3:
                f.write("\n./run_maml_%s_%d_%d.sh" %(e, i,j)) #Execute shell with the python command
            else:
                f.write("\n./run_meta_%s_%d_%d.sh" %(e, i,j)) #Execute shell with the python command
            f.close()

