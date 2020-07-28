# getting a PSF shape directly from the star data

1. fit a Moffat function, which is integrated over the fiber area, to the data. Then normalize the radial profiles: $$ r \rightarrow r / \mathrm{FWHM} $$ and $$ f \rightarrow f/A $$, where $$ \mathrm{FWHM} $$ and $$ A $$ are the best-fit FWHM and amplitude of the integrated Moffat function. Take a biweight location in small $$r /\mathrm{FWHM} $$ bins. 
	* get_psf_integrated.py is the code for this
	* PSF_runbiw.dat contains the result
	* PSF_integrated.png,  PSF_integrated_Moffat_comp.png, and PSF_direct_int_normalized.png show the results in plots

2. now that we have a general shape of the PSF, we can normalize it so that it goes to one at $$ r=0 $$, and so that $$ f(r/\mathrm{FWHM} = 0.5) = 0.5 $$. We fit this function, adjusting the width and amplitude as free parameters, to the star radial profiles, normalize these, and get another running biweight (biweight in small $$ r/\mathrm{FWHM} $$ bins.
	* get_psf_int_iterate.py is the code for this
	* PSF.tab contains the result
	* PSF_int_iterated_comp_2.png, PSF_integrated_iterated_2.png, and PSF_integrated_iterated_Moffat_comp_2.png show plots of the result.

3. at first, I didn't adjust the width in the intermediate result, so that we had $$ f(r/\mathrm{FWHM} = 0.5) \approx 0.4 $$. This lead to a slightly different result, which we are going to discard.
	* PSF_runbiw.dat contains the (not so good) result
	* PSF_integrated_iterated_Moffat_comp.png, PSF_integrated_iterated.png, and PSF_int_iterated_comp.png show this. 
