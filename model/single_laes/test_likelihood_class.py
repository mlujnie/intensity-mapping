import save_laes as ml
import glob
import numpy as np
import sys


bello = ml.SaveLike()
print("Initializing worked.")

lae_ids = [#2101630721,
2101630773,
2101630838,
2101630898,
2101630991,
2101631012,
2101631033,
2101631048,
2101631109,
2101631122,
2101631158,
2101631234,
2101631301,
2101631322,
2101631339,
2101631410,
2101631442,
2101631443,
2101631494,
2101631507,
2101631569,
2101631606,
2101631666,
2101631687,
2101631738,
2101631755,
2101631804,
2101631834]

#bello.lae_id = lae_id
#bello.load_from_shot()


print("flux: ", bello.starflux.dtype)
print("err: ", bello.starsigma.dtype)
print("dist: ", bello.stardists.dtype)
