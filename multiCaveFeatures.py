########################################################################################
## BASIC LIBRRAIES
import os
from os import path as p
import pandas as pd
import argpass
import multiprocessing
import glob
from tqdm import tqdm
import yaml


## SPECIFIC CAVEFEATURES FUNCTIONS ##
from modules_multiCaveFeatures import *
from pdbUtils import *
########################################################################################
def main():
    # load user inputs
    inputDir, outDir, msmsDir, aminoAcidTable, genCofactorLabels = read_inputs()

    os.makedirs(outDir, exist_ok=True)
    # initialise amino acid data
    aminoAcidNames, aminoAcidProperties = initialiseAminoAcidInformation(aminoAcidTable)
    # get list of pdbFiles in pdbDir
    idList, pdbList = getPdbList(inputDir)
#    process_serial(pdbList=pdbList,
#                 outDir=outDir, 
#                 aminoAcidNames=aminoAcidNames,
#                 aminoAcidProperties=aminoAcidProperties,
#                 msmsDir=msmsDir,
#                 pdbDir=inputDir,
#                 genCofactorLabels = genCofactorLabels)
#    #   
    # Process pdbList using multiprocessing
    process_pdbs(pdbList=pdbList,
                  outDir=outDir, 
                  aminoAcidNames=aminoAcidNames,
                    aminoAcidProperties=aminoAcidProperties,
                    msmsDir=msmsDir,
                    pdbDir=inputDir,
                    genCofactorLabels = genCofactorLabels)
    print("Merging output csvs!")
    merge_results(outDir)
########################################################################################
def merge_csv_files(csvFiles, outFile):
    chunk_size = 10  # Adjust this based on available memory
    for i in range(0, len(csvFiles), chunk_size):
        chunkFiles = csvFiles[i:i + chunk_size]
        dfs = [pd.read_csv(csvFile, index_col="Unnamed: 0") for csvFile in chunkFiles]
        merged_df = pd.concat(dfs, axis=0)
        merged_df.to_csv(outFile, mode='a', header=not os.path.exists(outFile), index=True)
        for csvFile in chunkFiles:
            os.remove(csvFile)


###############################################################################################
def batch_list(inputList, nBatches):
    inLength = len(inputList)
    batchSize = inLength // nBatches
    remainder = inLength // nBatches

    batches = [inputList[i * batchSize + min(i, remainder):(i + 1) * batchSize + min(i + 1, remainder)]
                for i in range(nBatches)]
    return batches
########################################################################################
def merge_results(outDir):
    nCpus = multiprocessing.cpu_count()
    csvFiles = glob.glob(os.path.join(outDir, "*_features.csv"))


    csvBatches = batch_list(csvFiles,nCpus)

    outCsvs = [p.join(outDir,f"caveFeatures_batch{i}.csv") for i in range(1,len(csvBatches)+1)]

    with multiprocessing.Pool(processes=nCpus) as pool:
        jobs = []
        for csvFiles, outCsv in zip(csvBatches, outCsvs):
            job = pool.apply_async(merge_csv_files,(csvFiles,outCsv))
            jobs.append(job)
        for job in jobs:
            job.get()
    finalMergedCsv = p.join(outDir, "multiCaveFeatures.csv")
    merge_csv_files(outCsvs, finalMergedCsv)
########################################################################################
def read_inputs():
    ## create an argpass parser, read config file, snip off ".py" if on the end of file
    parser = argpass.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config=args.config
    ## Read config.yaml into a dictionary
    with open(config,"r") as yamlFile:
        config = yaml.safe_load(yamlFile) 
    inputDir = config["inputDir"]
    outDir = config["outDir"]
    msmsDir = config["msmsDir"]
    aminoAcidTable = config["aminoAcidTable"]
    if "genCofactorLabels" in config:
        genCofactorLabels = config["genCofactorLabels"]
    else:
        genCofactorLabels = False

    return inputDir, outDir, msmsDir, aminoAcidTable, genCofactorLabels
########################################################################################
def process_serial(pdbList, outDir, aminoAcidNames, aminoAcidProperties, msmsDir,pdbDir, genCofactorLabels):
    for pdbFile in pdbList:
        process_pdbs_worker(pdbFile, outDir, aminoAcidNames, aminoAcidProperties, msmsDir, pdbDir, genCofactorLabels)
########################################################################################
def process_pdbs_worker(pdbFile, outDir, aminoAcidNames, aminoAcidProperties, msmsDir, pdbDir, genCofactorLabels):
    proteinName = p.splitext(p.basename(pdbFile))[0]
    pdbDf = pdb2df(pdbFile)

    if genCofactorLabels:
        cofactorDf = pdbDf[~pdbDf["RES_NAME"].isin(aminoAcidNames)]
        pdbDf = pdbDf[pdbDf["RES_NAME"].isin(aminoAcidNames)]
        noCoPdb = p.join(outDir, f"{proteinName}_noCofactor.pdb")
        df2Pdb(pdbDf,noCoPdb)
        exteriorDf, coreDf = findCoreExterior(pdbFile=noCoPdb, pdbDf=pdbDf,
                                          proteinName=proteinName, msmsDir=msmsDir,
                                          outDir=outDir)
        caveDfs, pocketTags = gen_multi_cave_regions(outDir=outDir,
                                pdbFile=noCoPdb)
        labelsDf = generate_cofactor_labels(caveDfs, pocketTags, cofactorDf, proteinName)
        os.remove(noCoPdb)

    else:
        exteriorDf, coreDf = findCoreExterior(pdbFile=pdbFile, pdbDf=pdbDf,
                                            proteinName=proteinName, msmsDir=msmsDir,
                                            outDir=outDir)
        caveDfs, pocketTags = gen_multi_cave_regions(outDir=outDir,
                                    pdbFile=pdbFile)

    ## GET ELEMENT COUNTS FOR EACH REGION ##
    extElementCountDf = element_count_in_region(regionDf=exteriorDf,
                                           regionName="ext",
                                           proteinName=proteinName)
    coreElementCountDf = element_count_in_region(regionDf=coreDf,
                                           regionName="core",
                                           proteinName=proteinName)

    
    ## GET AMINO ACID COUNTS FOR EACH REGION ##
    extAACountDf = amino_acid_count_in_region(regionDf=exteriorDf,
                                              regionName="ext",
                                              proteinName=proteinName,
                                              aminoAcidNames=aminoAcidNames)
    coreAACountDf = amino_acid_count_in_region(regionDf=coreDf,
                                              regionName="core",
                                              proteinName=proteinName,
                                              aminoAcidNames=aminoAcidNames)

    ## CALCULATE AMINO ACID PROPERTIES FOR EACH REGION ##
    extPropertiesDf = calculate_amino_acid_properties_in_region(aaCountDf=extAACountDf,
                                                                aminoAcidNames=aminoAcidNames,
                                                                aminoAcidProperties=aminoAcidProperties,
                                                                proteinName=proteinName, 
                                                                regionName="ext")
    corePropertiesDf = calculate_amino_acid_properties_in_region(aaCountDf=coreAACountDf,
                                                                aminoAcidNames=aminoAcidNames,
                                                                aminoAcidProperties=aminoAcidProperties,
                                                                proteinName=proteinName, 
                                                                regionName="core")    
    ## GET ELEMENT AND AA COUNTS AND AA PROPERTIES FOR ALL CAVES
    for caveDf, pocketTag in zip(caveDfs, pocketTags):
        pocketName = f"{proteinName}_{pocketTag}"
        caveElementCountDf = element_count_in_region(regionDf=caveDf,
                                           regionName="cave",
                                           proteinName=pocketName)
        caveAACountDf = amino_acid_count_in_region(regionDf=caveDf,
                                              regionName="cave",
                                              proteinName=pocketName,
                                              aminoAcidNames=aminoAcidNames)
        cavePropertiesDf = calculate_amino_acid_properties_in_region(aaCountDf=caveAACountDf,
                                                                aminoAcidNames=aminoAcidNames,
                                                                aminoAcidProperties=aminoAcidProperties,
                                                                proteinName=pocketName, 
                                                                regionName="cave")
        ## MERGE FEATURESETS
        dfsToConcat = [extElementCountDf,coreElementCountDf,caveElementCountDf,
                    extAACountDf,coreAACountDf,caveAACountDf,
                    extPropertiesDf,corePropertiesDf,cavePropertiesDf]
        if genCofactorLabels:
            label = labelsDf.loc[:,pocketName]
            label.rename("Cofactor",inplace=True)
            dfsToConcat.append(label)

        ## SORT OUT INDEXES
        for df in dfsToConcat:
            df.index = [pocketName]

        featuresDf = pd.concat(dfsToConcat, axis=1)
        # Save featuresDf to a CSV file
        saveFile = p.join(outDir, f"{pocketName}_features.csv")
        featuresDf.to_csv(saveFile, index=True)

########################################################################################
def process_pdbs(pdbList, outDir, aminoAcidNames, aminoAcidProperties, msmsDir,pdbDir, genCofactorLabels):
    # Use multiprocessing to parallelize the processing of pdbList
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_pdbs_worker,
                     tqdm( [(pdbFile, outDir, aminoAcidNames, aminoAcidProperties, msmsDir, pdbDir, genCofactorLabels) for pdbFile in pdbList],
                     total = len(pdbList)))


########################################################################################
main()