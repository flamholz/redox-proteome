# redox-proteome

Though the term "proteome" is sometimes used to refer to the set of coding sequences in a genome, e.g. as in "[reference proteome](https://www.uniprot.org/help/reference_proteome)," here proteome will refer to the relative or absolute levels of expressed proteins. 

## Reference coding sequences

Reference coding sequences live in `data/genomes/` and were drawn from UniprotKB entries for *E. coli* (UP000000625_83333.xml), yeast (UP000002311_559292) and cyanobacteria (UP000001425_1111708) proteins. Full documentation is give in `data/genomes/reference_proteomes.csv`.

## Amino acid properties 
Amino acid carbon content and Z_C values are found in data/aa_nosc.csv. Z_C values are drawn from LaRowe & Amend, ISME J 2016 and were checked by manual calculation (TODO). Polar requirement and hydropathy values are drawn from Shenhav & Zeevi, Science 2020, who draw those values from Haig & Hurst J. Mol. Evol. 1991.
