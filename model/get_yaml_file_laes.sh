SHOTID=$1
NAME=$2


cd /work/05865/maja_n/stampede2/master/chains-laes
mkdir $NAME
cd $NAME
cp ../run_cobaya.slurm .
python3 /work/05865/maja_n/stampede2/master/intensity-mapping/model/get_yaml_file_laes.py -s $SHOTID -n $NAME
chmod +x cobaya_job.run

echo "sbatch run_cobaya.slurm"
