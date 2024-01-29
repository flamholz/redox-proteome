# redox-proteome

Though the term "proteome" is sometimes used to refer to the set of coding sequences in a genome, e.g. as in "[reference proteome](https://www.uniprot.org/help/reference_proteome)," here proteome will refer to the relative or absolute levels of expressed proteins. 

## Directories

* data/ contains source and derived data in a directory hierarchy
* munge/ contains scripts that pre-process data for analysis and plotting
* notebooks/ contains scripts and notebooks that perform analysis
* mathematica/ contains Mathematica notebooks
* models/ contains files that define models used in code 

## Reference coding sequences

Reference coding sequences live in `data/genomes/` and were drawn from UniProt entries for *E. coli* (UP000000625_83333.xml), yeast (UP000002311_559292) and cyanobacteria (UP000001425_1111708) proteins. Full documentation of the reference sequences used is give in `data/genomes/reference_proteomes.csv`.The iPython notebook `notebooks/00_PreprocessCodingSequences.ipynb` extracts these coding sequences and related metadata from UniProt XML files and calculates $Z_C$ values for each coding sequence. 

## Amino acid properties 

Amino acid molecular weights, carbon content and $Z_C$ values are found in data/aa_nosc.csv. $Z_C$ values are drawn from LaRowe & Amend, ISME J 2016 and were checked by manual calculation (TODO). Polar requirement and hydropathy values are drawn from Shenhav & Zeevi, Science 2020, who draw those values from Haig & Hurst J. Mol. Evol. 1991.

## Proteomics data

Proteome data was downloaded from the relevant references (as described in the methods section), reformatted and stored in `data/proteomes`.

## GTDB

Data from the Genome Taxonomy Database (GTDB) is drawn from version 207, downloaded from [https://gtdb.ecogenomic.org/downloads](https://gtdb.ecogenomic.org/downloads) and stored in `data/gtdb`. We are working with the $\approx 60,000$ representative genomes rather than the full set of genomes since this limits computational complexity and, further, these genomes are, where possible, drawn from type strains with a higher degree of completeness and better annotation. 

