from hetdex_api.extract import Extract
import argparse
import sys
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shot", type=str, help="Shotid.")
args = parser.parse_args(sys.argv[1:])

shot = args.shot

E = Extract()
E.load_shot(shot, survey="hdr2")

gmag_limit = 22.
radius = 50.
psf = E.model_psf(gmag_limit=gmag_limit, radius=radius)

hdr = fits.Header()
hdr["SHOT"] = shot
hdr["GMAG_LIMIT"] = gmag_limit
hdr["RADIUS"] = radius

hdu = fits.PrimaryHDU(psf[0], header=hdr)
hdu_x, hdu_y = fits.ImageHDU(psf[1]), fits.ImageHDU(psf[2])
hdul = fits.HDUList([hdu, hdu_x, hdu_y])

hdul.writeto("/data/05865/maja_n/im2d/psf_hdr2/"+shot+".fits", overwrite=True)
print("Wrote to /data/05865/maja_n/im2d/psf_hdr2/"+shot+".fits")