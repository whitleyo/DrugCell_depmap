# DepMap Public 23Q4

##############
## Overview ##
##############

This DepMap release contains data from CRISPR knockout screens from project Achilles, as well as genomic characterization data from the CCLE project.

###############
## Pipelines ##
###############

### Achilles

This Achilles dataset contains the results of genome-scale CRISPR knockout screens for Achilles (using Avana Cas9 and Humagne-CD Cas12 libraries) and Achilles combined with Sanger's Project SCORE (KY Cas9 library) screens.  The dataset was processed using the following steps:
- Sum raw readcounts by SequenceID and collapse pDNAs by median (removing sequences with fingerprinting issues, total reads below 500000 or less then 25 mean reads per guide)
- Normalize readcounts such that pDNA sequences have the same mode, and all other sequences have median of nonessentials that matches the median of nonessentials in the corresponding pDNA batch
- Remove intergenic/non-targeting controls, Cas9 guides with multiple alignments to a gene, guides with >5 genomic alignments, and sgRNAs with inconsistent signal (see DropReason of guide maps).
- NA sgRNAs with pDNA counts less than one millionth of the pDNA pool
- Calculate the mean reads per guide for each sequence. Sequences with more then 185 mean reads per guide are considered passing.
- Compute the log2-fold-change (LFC) from pDNA counts for each sequence. Collapse to a gene-level by taking the median of guides.
- Calculate the Null-Normalized Median Difference (NNMD) from the LFC using the following equation: (median(essentials) - medain(nonessentials)) / MAD(nonessentials).  Sequences must be below a threshold of -1.25 to be considered passing.
- Remove sequences that do not have a Pearson coefficient > .41 with at least one other sequence for the screen when looking at genes with the highest variance in gene effect across screens and libraries.
- Calculate the NNMD for each screen after averaging passing sequences. Screen must be below a threshold of -1.25 to be considered passing.
- NA out readcounts for pDNAs which correspond with a SNP in any guide in a vector for a given cell line (to correct ancestry bias), then recompute LFC.
- Compute the naive gene score by collapsing the passing LFC data to a screen x gene matrix.
- Prior to running Chronos, NA apparent outgrowths in readcounts.
- Run Chronos per library-screen type to generate screen-level gene effect scores, apply copy number correction, then scale such that median of common essentials is at -1.0 and median of nonessentials is at 0, and correct for screen quality. Concatenate gene effects from all libraries into a single ScreenGeneEffect matrix.
- Run Chronos jointly on all libraries-screen types to generate model-level gene effect scores, apply copy correction, scale, and correct screen quality as described above.  This produces the integrated CRISPRGeneEffect matrix using Chronos's innate batch correction.
- Using the CRISPRGeneEffect, identify pan-dependent genes as those for whom 90% of cell lines rank the gene above a given dependency cutoff. The cutoff is determined from the central minimum in a histogram of gene ranks in their 90th percentile least dependent line.
- For each Chronos gene effect score from both the ScreenGeneEffect and CRISPRGeneEffect, infer the probability that the score represents a true dependency. This is done using an EM step until convergence independently in each screen/model. The dependent distribution is given by the list of essential genes. The null distribution is determined from unexpressed gene scores in those cell lines that have expression data available, and from the nonessential gene list in the remainder.
The essential  and nonessential controls used throughout the analysis are the Hart reference nonessentials and the intersection of the Hart and Blomen essentials. See Hart et al., Mol. Syst. Biol, 2014 and Blomen et al., Science, 2015. They are provided with this release as AchillesCommonEssentialControls.csv and AchillesNonessentialControls.csv.

### Expression

DepMap expression data is quantified from RNAseq files using the GTEx pipelines. A detailed description of the pipelines and tool versions can be found here: https://github.com/broadinstitute/ccle_processing#rnaseq. We provide a subset of the data files outputted from this pipeline available on FireCloud. These are aligned to hg38.

### Copy number

DepMap WES copy number data is generated by running the GATK copy number pipeline aligned to hg38. Tutorials and descriptions of this method can be found here https://software.broadinstitute.org/gatk/documentation/article?id=11682, https://software.broadinstitute.org/gatk/documentation/article?id=11683. WES samples have been realigned to hg38 and run through this pipeline.

### Mutations

DepMap mutation calls are generated using Mutect2 and annotated and filtered downstream. Variants are aligned to hg38. Detailed documentation can be found here https://storage.googleapis.com/shared-portal-files/Tools/%5BDMC%20Communication%5D%2022Q4%20Mutation%20Pipeline%20Update.pdf.

### Fusions

DepMap generates RNAseq based fusion calls using the STAR-Fusion pipeline. A comprehensive overview of how the STAR-Fusion pipeline works can be found here: https://github.com/STAR-Fusion/STAR-Fusion/wiki. We run STAR-Fusion version 1.6.0 using the plug-n-play resources available in the STAR-Fusion docs for gencode v29. We run the fusion calling with default parameters except we add the --no_annotation_filter and --min_FFPM 0 arguments to prevent filtering.

###########
## Files ##
###########

### README.txt

Description of all files contained in this release

### ScreenGeneEffect.csv

Pipeline: Achilles

_Post-Chronos_
Gene effect estimates for all screens Chronos processed by library, copy number corrected, scaled, screen quality corrected then concatenated.
- Columns: Gene - Rows: ScreenID

### ScreenGeneEffectUncorrected.csv

Pipeline: Achilles

_Post-Chronos_
Gene effect estimates for all screens Chronos processed by library, then concatenated. No copy number correction or scaling.
- Columns: Gene - Rows: ScreenID

### ScreenNaiveGeneScore.csv

Pipeline: Achilles

_Post-Chronos_
LFC collapsed by mean of sequences and median of guides, computed per library-screen type then concatenated.
- Columns: Gene - Rows: ScreenID

### ScreenGeneDependency.csv

Pipeline: Achilles

_Post-Chronos_
Gene dependency probability estimates for all screens Chronos processed by library-screen type, then concatenated.
- Columns: Gene - Rows: ScreenID

### CRISPRScreenMap.csv

Pipeline: Achilles

_Post-Chronos_
Map from ModelID to all ScreenIDs combined to make up a given model's data in the CRISPRGeneEffect matrix.
Columns: - ModelID - ScreenID

### CRISPRGeneEffect.csv

Pipeline: Achilles

_Post-Chronos_
Gene effect estimates for all models, integrated using Chronos.  Copy number corrected, scaled, and screen quality corrected.
- Columns: Gene - Rows: ModelID

### CRISPRGeneEffectUncorrected.csv

Pipeline: Achilles

_Post-Chronos_
Gene effect estimates for all models, integrated using Chronos. No copy number correction or scaling.
- Columns: Gene - Rows: ModelID

### CRISPRGeneDependency.csv

Pipeline: Achilles

_Post-Chronos_
Gene dependency probability estimates for all models in the integrated gene effect.
- Columns: Gene - Rows: ModelID

### CRISPRInferredGuideEfficacy.csv

Pipeline: Achilles

_Post-Chronos_
The estimates for the efficacies of all reagents in the different libraries-screen types, computed from the Chronos runs.
Columns: - sgRNA - Efficacy

### CRISPRInferredLibraryEffect.csv

Pipeline: Achilles

_Post-Chronos_
The estimates for the library batch effects identified by Chronos.
- Columns: Gene - Rows: LibraryBatch

### CRISPRInitialOffset.csv

Pipeline: Achilles

_Post-Chronos_
Estimated log fold pDNA error for each sgrna in each library, as identified by Chronos.
- Columns: pDNABatch - Rows: sgRNA

### CRISPRInferredModelGrowthRate.csv

Pipeline: Achilles

_Post-Chronos_
The estimates for the growth rate of all models in the different libraries-screen types, computed from the Chronos runs.
Columns: - ScreenID - Achilles-Avana-2D - Achilles-Humagne-CD-2D - Achilles-Humagne-CD-3D - Project-Score-KY

### CRISPRInferredModelEfficacy.csv

Pipeline: Achilles

_Post-Chronos_
The estimates for the efficacy of all models in the different libraries-screen types, computed from the Chronos runs.
Columns: - ModelID - Achilles-Avana-2D - Achilles-Humagne-CD-2D - Achilles-Humagne-CD-3D - Project-Score-KY

### CRISPRInferredSequenceOverdispersion.csv

Pipeline: Achilles

_Post-Chronos_
List of genes identified as dependencies across all lines. Each entry is separated by a newline.

### CRISPRInitialOffset.csv

Pipeline: Achilles

_Post-Chronos_
The log fold error in pDNA estimated by Chronos, per library
- Columns: LibraryBatch - Rows: sgRNA

### CRISPRInferredCommonEssentials.csv

Pipeline: Achilles

_Post-Chronos_
List of genes identified as dependencies across all lines. Each entry is separated by a newline.

### AchillesCommonEssentialControls.csv

Pipeline: Achilles

_Pre-Chronos file_

List of genes used as positive controls, intersection of Biomen (2014) and Hart (2015) essentials. Each entry is separated by a newline.
The scores of these genes are used as the dependent distribution for inferring dependency probability.

### AchillesNonessentialControls.csv

Pipeline: Achilles

_Pre-Chronos file_

List of genes used as negative controls (Hart (2014) nonessentials). Each entry is separated by a newline.

### AvanaRawReadcounts.csv

Pipeline: Achilles

_Pre-Chronos file_
Summed guide-level read counts for each sequence screened with the Avana Cas9 library.
-Columns: SequenceID - Rows: sgRNA

### HumagneRawReadcounts.csv

Pipeline: Achilles

_Pre-Chronos file_
Summed guide-level read counts for each sequence screened with the Humagne-CD Cas12 library.
-Columns: SequenceID - Rows: sgRNA

### KYRawReadcounts.csv

Pipeline: Achilles

_Pre-Chronos file_
Summed guide-level read counts for each sequence screened with the Sanger's KY Cas9 library.
-Columns: SequenceID - Rows: sgRNA

### AvanaLogfoldChange.csv

Pipeline: Achilles

_Pre-Chronos file_
Log2-fold-change from pDNA counts for each sequence screened with the Avana Cas9 Library.
-Columns: SequenceID - Rows: sgRNA

### HumagneLogfoldChange.csv

Pipeline: Achilles

_Pre-Chronos file_
Log2-fold-change from pDNA counts for each sequence screened with the Humagne-CD Library.
-Columns: SequenceID - Rows: sgRNA

### KYLogfoldChange.csv

Pipeline: Achilles

_Pre-Chronos file_
Log2-fold-change from pDNA counts for each sequence screened with the Sanger's KY Cas9 Library.
-Columns: SequenceID - Rows: sgRNA

### AvanaGuideMap.csv

Pipeline: Achilles

_Pre-Chronos file_
Mapping of sgRNAs to Genes in the Avana Cas9 library.
Columns: - sgRNA: guide in vector - GenomeAlignment: alignment to hg38 - Gene: HUGO (entrez) - nAlignments: total number of alignments for a given sgRNA - UsedByChronos: boolean indicating if sgRNA was included in Chronos analysis - DropReason: why a guide was removed prior to Chronos analysis

### HumagneGuideMap.csv

Pipeline: Achilles

_Pre-Chronos file_

Mapping of sgRNAs to Genes in the Humagne-CD Cas12 library.

Columns:
- sgRNA: first guide in vector
- GenomeAlignment: alignment of first guide to hg38
- sgRNA2: second guide in vector
- GenomeAlignment2: alignment of second guide to hg38
- Set: which half of the library the vector is part of (C, D, or Both)
- Gene: HUGO (entrez)
- nAlignments: total number of alignments for a given sgRNA
- UsedByChronos: boolean indicating if sgRNA was included in Chronos analysis
- DropReason: why a guide was removed prior to Chronos analysis


### KYGuideMap.csv

Pipeline: Achilles

_Pre-Chronos file_
Mapping of sgRNAs to Genes in the Sanger's KY Cas9 library.
Columns: - sgRNA: guide in vector - GenomeAlignment: alignment to hg38 - Gene: HUGO (entrez) - nAlignments: total number of alignments for a given sgRNA - UsedByChronos: boolean indicating if sgRNA was included in Chronos analysis - DropReason: why a guide was removed prior to Chronos analysis

### ScreenSequenceMap.csv

Pipeline: Achilles

_Pre-Chronos file_
Mapping of SequenceIDs to ScreenID and related info.
Columns: - SequenceID - ScreenID - ModelConditionID - ModelID - ScreenType: 2DS = 2D standard - Library - Days - pDNABatch - PassesQC - ExcludeFromCRISPRCombined

### AchillesScreenQCReport.csv

Pipeline: Achilles

_Pre-Chronos file_

Screen-level quality control metrics.

Columns:
- ScreenID
- ScreenNNMD: null-normalized median difference (threshold of -1.25)
- ScreenROCAUC: area under the Receiver Operating Characteristic curve 
for essential (positive) vs nonessentials (negative) controls
- ScreenFPR: false positive rate, computed as fraction of nonessentials in
the 15th percentile most depleted genes
- ScreenMeanEssentialDepletion: mean LFC of essentials
- ScreenMeanNonessentialDepletion: mean LFC of nonessentials
- ScreenSTDEssentials: standard deviation of essential gene LFCs
- ScreenSTDNonessentials: standard deviation of nonessential gene LFCs
- PassesQC: boolean indicating if screen passes QC thresholds
- CanInclude: boolean indicating if screen can be included in dataset
- QCStatus: string describing QC status
- HasCopyNumber: boolean indicating if screen has copy number data
- ModelConditionID
- ModelID
- Library
- ScreenType
- CasActivity: percentage of cells remaining GFP negative on days 12-14 of cas9
  activity assay as measured by FACs
- ScreenDoublingTime: hours for cells to double
- nPassingSequences: number of sequences that pass sequence-level QC
- nIncludedSequences: number of sequences that can be included


### AchillesSequenceQCReport.csv

Pipeline: Achilles

_Pre-Chronos file_

Sequence-level quality control metrics.

Columns:
- SequenceID
- SequenceTotalReads: total number of raw readcounts
- SequenceMeanReadsPerGuide: mean reads per guide following normalization (threshold of 185)
- SequenceNNMD: null-normalized median difference (threshold of -1.25)
- PassesQC: boolean indicating if sequence passes QC thresholds
- CanInclude: boolean indicating if sequence can be included in dataset
- QCStatus: string describing QC status
- SequenceMaxCorr: max correlation of high variance genes with other 
sequences from same screen (threshold of 0.41)
- ScreenID
- ScreenPassesQC: boolean indicating if corresponding screen passes QC thresholds


### AchillesHighVarianceGeneControls.csv

Pipeline: Achilles

_Pre-Chronos file_
List of genes with variable gene effects across models and libraries. Used for sequence correlation in QC.

### OmicsExpressionGenesExpectedCountProfile.csv

Pipeline: Expression

RNAseq read count data from RSEM.

- Rows: Omics Profile IDs
- Columns: genes (HGNC symbol and Ensembl ID)

### OmicsExpressionProteinCodingGenesTPMLogp1.csv

Pipeline: Expression

Gene expression TPM values of the protein coding genes for DepMap cell lines. Values are inferred from RNA-seq data using the RSEM tool and are reported after log2 transformation, using a pseudo-count of 1; log2(TPM+1).
Additional RNA-seq-based expression measurements are available for download as part of the full DepMap Data Release
More information on the DepMap Omics processing pipeline is available at <https://github.com/broadinstitute/depmap_omics>.

### OmicsExpressionTranscriptsExpectedCountProfile.csv

Pipeline: Expression

RNAseq read count data from RSEM.

- Rows: Omics Profile IDs
- Columns: transcripts (HGNC symbol and ensembl transcript ID)

### OmicsExpressionTranscriptsTPMLogp1Profile.csv

Pipeline: Expression

RNAseq transcript tpm data using RSEM. Log2 transformed, using a pseudo-count of 1.
- Rows: Omics Profile IDs
- Columns: transcripts (HGNC symbol and ensembl transcript ID)

### OmicsExpressionAllGenesTPMLogp1Profile.csv

Pipeline: Expression

Gene expression TPM values of all genes for DepMap cell lines. Values are inferred from RNA-seq data using the RSEM tool and are reported after log2 transformation, using a pseudo-count of 1; log2(TPM+1).
Additional RNA-seq-based expression measurements are available for download as part of the full DepMap Data Release
More information on the DepMap Omics processing pipeline is available at <https://github.com/broadinstitute/depmap_omics>.
- Rows: Omics Profile IDs
- Columns: Gene (HGNC symbol and ensembl gene ID)

### OmicsExpressionAllGenesEffectiveLengthProfile.csv

Pipeline: Expression

Gene effective length for all genes output from RSEM
- Rows: Omics Profile IDs
- Columns: Gene (HGNC symbol and ensembl gene ID)

### OmicsCNSegmentsProfile.csv

Pipeline: Copy number


Segment level copy number data
- Profile ID
- Chromosome
- Start (bp start of the segment)
- End (bp end of the segment)
- Num_Probes (the number of targeting probes that make up this segment)
- Segment_Mean (relative copy ratio for that segment)
- amplification status (+,-,0)

### OmicsCNGene.csv

Pipeline: Copy number

Gene-level copy number data that is log2 transformed with a pseudo-count of 1; log2(CN ratio + 1) . Inferred from WGS, WES or SNP array depending on the availability of the data type. Values are calculated by mapping genes onto the segment level calls and computing a weighted average along the genomic coordinate.
Genes that overlap with segmental duplication regions and/or flagged by repeatMasker are masked in this matrix. For details see https://github.com/broadinstitute/depmap_omics/blob/master/docs/source/dna.md#masking .
Additional copy number datasets are available for download as part of the full DepMap Data Release.
More information on the DepMap Omics processing pipeline is available at <https://github.com/broadinstitute/depmap_omics>.

### OmicsFusionFiltered.csv

Pipeline: Fusions

Gene fusion data derived from RNAseq data. Data is filtered using by performing the following:

- Removing fusion involving mitochondrial chromosomes or HLA genes
- Removed common false positive fusions (red herring annotations as described in the STAR-Fusion docs)
- Recurrent fusions observed in CCLE across cell lines (in 10% or more of the samples)
- Removed fusions where SpliceType='INCL_NON_REF_SPLICE' and LargeAnchorSupport='NO_LDAS' and FFPM < 0.1
- FFPM < 0.05
Column descriptions can be found in the STAR-Fusion wiki, except for CCLE_count, which indicates the number of CCLE samples that have this fusion.

### OmicsFusionUnfilteredProfile.csv

Pipeline: Fusions

Gene fusion data derived from RNAseq data. Data is unfiltered. Column descriptions can be found in the STAR-Fusion wiki.
Samples are identified by Profile IDs.

### OmicsSomaticMutations.csv

Pipeline: Mutations

MAF-like file containing information on all the somatic point mutations and indels called in the DepMap cell lines. The calls are generated from Mutect2.

Additional processed mutation matrices containing genotyped mutation calls are available for download as part of the full DepMap Data Release.

Columns:

- Chrom
- Pos
- Ref
- Alt
- AF
- DP
- RefCount
- AltCount
- GT
- PS
- VariantType
- VariantInfo
- DNAChange
- ProteinChange
- HugoSymbol
- EnsemblGeneID
- EnsemblFeatureID
- HgncName
- HgncFamily
- UniprotID
- DbsnpRsID
- GcContent
- LofGeneName
- LofGeneId
- LofNumberOfTranscriptsInGene
- LofPercentOfTranscriptsAffected
- NMD
- MolecularConsequence
- VepImpact
- VepBiotype
- VepHgncID
- VepExistingVariation
- VepManeSelect
- VepENSP
- VepSwissprot
- Sift
- Polyphen
- GnomadeAF
- GnomadgAF
- VepClinSig
- VepSomatic
- VepPliGeneValue
- VepLofTool
- OncogeneHighImpact
- TumorSuppressorHighImpact
- TranscriptLikelyLof
- Brca1FuncScore
- CivicID
- CivicDescription
- CivicScore
- LikelyLoF
- HessDriver
- HessSignature
- RevelScore
- PharmgkbId
- DidaID
- DidaName
- GwasDisease
- GwasPmID
- GtexGene
- ProveanPrediction
- Rescue
- EntrezGeneID
- ModelID

For details, see https://storage.googleapis.com/shared-portal-files/Tools/23Q4_Mutation_Pipeline_Documentation.pdf


### OmicsSomaticMutationsProfile.csv

Pipeline: Mutations

MAF-like formatted file containing information on all the somatic point mutations and indels called in the DepMap cell lines. The calls are generated from Mutect2.

Additional processed mutation matrices containing genotyped mutation calls are available for download as part of the full DepMap Data Release.

Samples are identified by Profile IDs.

Columns:

- Chrom
- Pos
- Ref
- Alt
- AF
- DP
- RefCount
- AltCount
- GT
- PS
- VariantType
- VariantInfo
- DNAChange
- ProteinChange
- HugoSymbol
- EnsemblGeneID
- EnsemblFeatureID
- HgncName
- HgncFamily
- UniprotID
- DbsnpRsID
- GcContent
- LofGeneName
- LofGeneId
- LofNumberOfTranscriptsInGene
- LofPercentOfTranscriptsAffected
- NMD
- MolecularConsequence
- VepImpact
- VepBiotype
- VepHgncID
- VepExistingVariation
- VepManeSelect
- VepENSP
- VepSwissprot
- Sift
- Polyphen
- GnomadeAF
- GnomadgAF
- VepClinSig
- VepSomatic
- VepPliGeneValue
- VepLofTool
- OncogeneHighImpact
- TumorSuppressorHighImpact
- TranscriptLikelyLof
- Brca1FuncScore
- CivicID
- CivicDescription
- CivicScore
- LikelyLoF
- HessDriver
- HessSignature
- RevelScore
- PharmgkbId
- DidaID
- DidaName
- GwasDisease
- GwasPmID
- GtexGene
- ProveanPrediction
- Rescue
- EntrezGeneID
- ProfileID

For details, see https://storage.googleapis.com/shared-portal-files/Tools/23Q4_Mutation_Pipeline_Documentation.pdf


### OmicsSomaticMutationsMAFProfile.maf

Pipeline: Mutations

MAF file containing information on all the somatic point mutations and indels called in the DepMap cell lines. The calls are generated from Mutect2. A description of the various columns is in the DepMap Release README file.
Additional processed mutation matrices containing genotyped mutation calls are available for download as part of the full DepMap Data Release.
This file contains the same variants as OmicsSomaticMutationsProfile.csv, but follows the standard MAF format suitable for downstream analysis tools such as maftools.
Samples are identified by Profile IDs.

### OmicsSomaticMutationsMatrixHotspot.csv

Pipeline: Mutations

Genotyped matrix determining for each cell line whether each gene has at least one hot spot mutation.
A variant is considered a hot spot if it's present in one of the following: Hess et al. 2019 paper, OncoKB hotspot, COSMIC mutation significance tier 1.
(0 == no mutation; If there is one or more hot spot mutations in the same gene for the same cell line, the allele frequencies are summed, and if the sum is greater than 0.95, a value of 2 is assigned and if not, a value of 1 is assigned.)

### OmicsSomaticMutationsMatrixDamaging.csv

Pipeline: Mutations

Genotyped matrix determining for each cell line whether each gene has at least one damaging mutation.
A variant is considered a damaging mutation if LikelyLoF == True.
(0 == no mutation; If there is one or more damaging mutations in the same gene for the same cell line, the allele frequencies are summed, and if the sum is greater than 0.95, a value of 2 is assigned and if not, a value of 1 is assigned.)

### OmicsExpressionGeneSetEnrichment.csv

Pipeline: Expression

Single Sample Gene Set Enrichment Analysis
Scores were calculated using Single Sample Gene Set Enrichment Analysis
based on the z-scores of the log2(tpm + 1) of gene-level expression data.
Details about the R script used to run the analysis
can be found here: https://github.com/broadinstitute/ssGSEA2.0
- Columns: gene set enrichment pathways
- Rows: Model IDs

### OmicsExpressionGeneSetEnrichmentProfile.csv

Pipeline: Expression

Single Sample Gene Set Enrichment Analysis
Scores were calculated using Single Sample Gene Set Enrichment Analysis
based on the z-scores of the log2(tpm + 1) of gene-level expression data.
Details about the R script used to run the analysis
can be found here: https://github.com/broadinstitute/ssGSEA2.0
- Columns: gene set enrichment pathways
- Rows: Omics Profile IDs

### OmicsProfiles.csv

Profile ID mapping information
Columns: ProfileID, ModelID, ModelConditionID, Datatype, and WESKit
Rows: ProfileID

### OmicsDefaultModelProfiles.csv

indicates which profiles are selected in the model-level datasets
Columns: ModelID, ProfileID, and ProfileType (dna/rna)

### OmicsDefaultModelConditionProfiles.csv

indicates which profiles are selected for each model condition in Achilles postprocessing
Columns: ModelConditionID, ProfileID, and ProfileType (dna/rna)

### OmicsGuideMutationsBinaryKY.csv

binary matrix indicating whether there are mutations in guide locations from the KY library.
KY guide library (same as the one used in project Score) can be accessed from https://score.depmap.sanger.ac.uk/downloads.
Columns: chrom, start, end, sgRNA, and ModelIDs

### OmicsGuideMutationsBinaryHumagne.csv

binary matrix indicating whether there are mutations in guide locations from the Humangne library.
Humagne guide library can be accessed from Addgene (https://www.addgene.org/pooled-library/broadgpp-human-knockout-humagne/).
Columns: chrom, start, end, sgRNA, and ModelIDs

### OmicsGuideMutationsBinaryAvana.csv

binary matrix indicating whether there are mutations in guide locations from the Avana library.
Avana guide library can be accessed from AvanaGuideMap.csv.
Columns: chrom, start, end, sgRNA, and ModelIDs

### Model.csv

Metadata describing all cancer models/cell lines which are referenced by a dataset contained within the DepMap portal.
Columns:
  - ModelID: Unique identifier for the model
  - PatientID: Unique identifier for models derived from the same tissue sample
  - CellLineName: Commonly used cell line name
  - StrippedCellLineName: Commonly used cell line name without characters or spaces
  - DepmapModelType: Abbreviated ID for model type. For cancer models, this field is from Oncotree, information for other disease types are generated by DepMap
  - OncotreeLineage: Lineage of model. For cancer models, this field is from Oncotree, information for other disease types are generated by DepMap
  - OncotreePrimaryDisease: Primary disease of model. For cancer models, this field is from Oncotree, information for other disease types are generated by DepMap
  - OncotreeSubtype: Subtype of model. For cancer models, this field is from Oncotree, information for other disease types are generated by DepMap
  - OncotreeCode: For cancer models, this field is based on Oncotree. For some models for which no corresponding code exists, this field is left blank
  - LegacyMolecularSubtype: Additional information about model molecular features
  - PatientMolecularSubtype: Aggregates information known about the patient tumor
  - RRID: Cellosaurus ID
  - Age: Age at time of sampling
  - AgeCategory: Age category at time of sampling (Adult, Pediatric, Fetus, Unknown)
  - Sex: Sex at time of sampling (Female, Male, Unknown)
  - PatientRace: Patient/clinical indicated race (not derived)
  - PrimaryOrMetastasis: Site of the primary tumor where cancer originated from (Primary, Metastatic, Recurrance, Other, Unknown)
  - SampleCollectionSite: Site of tissue sample collection
  - SourceType: Indicates where model was onbaorded from (Commerical, Academic lab, Other)
  - SourceDetail: Details on where model was onboarded from
  - TreatmentStatus: Indicates if sample was collected before, during, or after the patient's cancer treatment (Pre-treatment, Active treatment, Post-treatment, Unknown)
  - TreatmentDetails: Details about patient treatment
  - GrowthPattern: Format model onboarded in (Adherent, Suspension, Organoid, Neurosphere, Neurosphere 2D, Unknown)
  - OnboardedMedia: Description of onboarding media
  - FormulationID: The unique identifier of the onboarding media 
  - EngineeredModel: Indicates if model was engineered (genetic knockout, cultured to resistance, other)
  - TissueOrigin: Indicates tissue model was derived from (Human, Mouse, Other)
  - CCLEName: CCLE name for the cell line
  - CatalogNumber: Catalogue number of cell model, if commercial
  - PlateCoating: Coating on plate model onboarded in (Laminin, Matrigel, Collagen, None)
  - ModelDerivationMaterial: Indicates what material a model was derived from (Fresh tissue, PDX, Other)
  - PublicComments: Comments released to portals
  - WTSIMasterCellID: WTSI ID
  - SangerModelID: Sanger ID
  - COSMICID: Cosmic ID
  - LegacySubSubtype: None


### ModelCondition.csv

The conditions under which the model was assayed.
Columns:
  - ModelConditionID: Unique identifier for each model condition. 
  - ModelID: Unique identifier for each model (same ID as in model table)
  - ParentModelConditionID: ID of parental model condition for new model conditions derived from other model conditions
  - CellFormat: Format the cell line is being grown in (Adherent, Suspension, Mixed, Organoid, Neurosphere, Unknown). This can be different than Growth Pattern.
  - PassageNumber: Number of cell line passages (<5, 6-10, 10+)
  - GrowthMedia: Media condition was grown in at the model condition level
  - FormulationID: Media formulation
  - PlateCoating: Substrate used to coat plates (Laminin, Matrigel, None, Unknown)


