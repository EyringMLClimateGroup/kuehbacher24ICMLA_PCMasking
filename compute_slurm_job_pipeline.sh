#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=con_flnt_lats_lon-120
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --output=LOG.con_flnt_lats_lon-120-%j.o
#SBATCH --error=LOG.con_flnt_lats_lon-120-%j.o
##SBATCH --exclusive
#SBATCH --mail-type=FAIL
#SBATCH --time=08:00:00


# Paths
scriptPath=`pwd`
logPath=${scriptPath}/logs

# Scripts
pyScript=pipeline
cfgFile=cfg_${pyScript}.yml

## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

cd $scriptPath

logFile=`ls LOG.con_flnt_lats_lon-120-*`
echo $scriptPath/$logFile $logPath/$logFile
cat $scriptPath/$cfgFile > $logFile

if [ ! -f ${scriptPath}/${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    module load python3/unstable
    jupyter nbconvert --to script ${pyScript}.ipynb
    module purge
    echo ""
fi

echo "Run PCMCI"
source /pf/b/b309172/.bashrc
conda activate tg38plus
python ${pyScript}.py -c $cfgFile

# Clean-up
rm ${pyScript}.py
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
mv $scriptPath/$logFile $logPath/$logFile

echo ""
echo "---------- Finished $0 ----------"