SHOTID=$1
NAME=$2


cd /data/05865/maja_n/cobaya-chains
mkdir $NAME
cd $NAME
cp ../run_cobaya.slurm .
python /data/05865/maja_n/intensity-mapping/model/get_yaml_file.py -s $SHOTID -n $NAME
chmod +x cobaya_job.run

echo "sbatch run_cobaya.slurm"
