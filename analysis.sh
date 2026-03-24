punlearn specextract
for i in 0 1 2 3 4; 
    do specextract mode=h infile="output/source_$i.fits[sky=region(output/overall.reg)]" \
            outroot="spec_out/source_$i" \
            asp="3392/repro/pcadf03392_001N001_asol1.fits" \
            badpixfile="3392/repro/acisf03392_repro_bpix1.fits" \
            mskfile="3392/repro/acisf03392_001N004_msk1.fits" \
            clobber=yes energy_wmap=1500:8000; done
