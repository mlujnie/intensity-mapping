SHOTID=$1
NAME=$2


cd /work/05865/maja_n/stampede2/master/cobaya-chains
mkdir psf_int_moffat_$NAME
cd psf_int_moffat_$NAME
cp ../run_cobaya.slurm .
python3 /work/05865/maja_n/stampede2/master/intensity-mapping/model/get_psf/integrated/get_yaml_file.py -s $SHOTID -n $NAME
chmod +x cobaya_job.run

echo "sbatch run_cobaya.slurm"
