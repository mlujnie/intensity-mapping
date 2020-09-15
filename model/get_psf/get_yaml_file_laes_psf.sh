SHOTID=$1
NAME=$2


cd /work/05865/maja_n/stampede2/master/chains-laes
mkdir psf_$NAME
cd psf_$NAME
cp ../run_cobaya.slurm .
cp /work/05865/maja_n/stampede2/master/intensity-mapping/model/get_psf/fwhm_prior.py .
cp /work/05865/maja_n/stampede2/master/fwhm_posteriors/fwhm_${SHOTID}.dat fwhm.dat
python3 /work/05865/maja_n/stampede2/master/intensity-mapping/model/get_psf/get_yaml_file_laes_psf.py -s $SHOTID -n $NAME
chmod +x cobaya_job.run

echo "sbatch run_cobaya.slurm"
