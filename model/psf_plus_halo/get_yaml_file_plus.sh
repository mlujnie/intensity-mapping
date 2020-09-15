NAME=$1

cd /work/05865/maja_n/stampede2/master/chains-laes
mkdir plus_$NAME
cd plus_$NAME
cp ../run_cobaya.slurm .
python3 /work/05865/maja_n/stampede2/master/intensity-mapping/model/psf_plus_halo/get_yaml_file_plus_2.py -n $NAME
chmod +x cobaya_job.run

echo "sbatch run_cobaya.slurm"
