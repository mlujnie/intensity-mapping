SHOTID=$1

cd /work/05865/maja_n/stampede2/master/radial_profiles
mkdir stars_$SHOTID
cd /work/05865/maja_n/stampede2/master/intensity-mapping/model/prepare_stars
python3 get_star_data.py -s $SHOTID
pwd
cd -
echo "Finished."
