SHOTID=$1

cd /data/05865/maja_n/radial_profiles
mkdir stars_$SHOTID
cd /data/05865/maja_n/intensity-mapping/model
python3 get_star_data.py -s $SHOTID
cd -
echo "Finished."
