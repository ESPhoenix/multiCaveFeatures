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
def read_inputs():
    ## create an argpass parser, read config file, snip off ".py" if on the end of file
    parser = argpass.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config=args.config
    ## Read config.yaml into a dictionary
    with open(config,"r") as yamlFile:
        config = yaml.safe_load(yamlFile) 

    pathInfo = config["pathInfo"]

    optionsInfo = config["optionsInfo"]

    return pathInfo, optionsInfo
########################################################################################
def main():
    # load user inputs
    pathInfo, optionsInfo= read_inputs()

    os.makedirs(pathInfo["outDir"], exist_ok=True)
    # initialise amino acid data
    # get list of pdbFiles in pdbDir
    idList, pdbList = getPdbList(pathInfo["inputDir"])
    # process_serial(pdbList=pdbList, pathInfo = pathInfo, optionsInfo = optionsInfo)
      
    # Process pdbList using multiprocessing
    process_pdbs(pdbList=pdbList, pathInfo = pathInfo, optionsInfo = optionsInfo)
    print("Merging output csvs!")
    merge_results(pathInfo["outDir"])
########################################################################################
def process_serial(pdbList, pathInfo, optionsInfo):

    ## get AA names and properties (do outside loop to save time)
    aminoAcidNames, aminoAcidProperties = initialiseAminoAcidInformation(pathInfo["aminoAcidTable"])
    ## run worker in serial (slow but easier to debug!)
    for pdbFile in pdbList:
        process_pdbs_worker(pdbFile, pathInfo, optionsInfo, aminoAcidNames, aminoAcidProperties)

########################################################################################
def process_pdbs(pdbList, pathInfo, optionsInfo):
    ## get AA names and properties (do outside mp to save time)
    aminoAcidNames, aminoAcidProperties = initialiseAminoAcidInformation(pathInfo["aminoAcidTable"])
    # Use multiprocessing to parallelize the processing of pdbList
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_pdbs_worker,
                     tqdm( [(pdbFile, pathInfo, optionsInfo, aminoAcidNames, aminoAcidProperties) for pdbFile in pdbList],
                     total = len(pdbList)))
########################################################################################
def process_pdbs_worker(pdbFile, pathInfo, optionsInfo, aminoAcidNames, aminoAcidProperties):
    ## read pathInfo to get key paths
    outDir = pathInfo["outDir"]
    msmsDir = pathInfo["msmsDir"]
    ## get protName for index and load pdb file as a dataframe
    proteinName = p.splitext(p.basename(pdbFile))[0]
    pdbDf = pdb2df(pdbFile)
    # if instructed to generate cofactor labels for classifier
    if optionsInfo["genCofactorLabels"]:
        # remove cofactor related rows from dataframe
        cofactorDf = pdbDf[~pdbDf["RES_NAME"].isin(aminoAcidNames)]
        pdbDf = pdbDf[pdbDf["RES_NAME"].isin(aminoAcidNames)]
        noCoPdb = p.join(outDir, f"{proteinName}_noCofactor.pdb")
        # write no-cofactor dataframe to pdb file
        df2Pdb(pdbDf,noCoPdb)
        # use MSMS to split into core and exterior dataframes
        exteriorDf, coreDf = findCoreExterior(pdbFile=noCoPdb, pdbDf=pdbDf,
                                          proteinName=proteinName, msmsDir=msmsDir,
                                          outDir=outDir)
        # use fpocket to identify caves 
        caveDfs, pocketTags, fpocketFeatures = gen_multi_cave_regions(outDir=outDir,
                                pdbFile=noCoPdb)
        # if not caves found, skip this protein entirely
        if caveDfs == None or len(pocketTags) == 0:
            return
        # label each row as cofactor binding or vacant 
        labelsDf = generate_cofactor_labels(caveDfs, pocketTags, cofactorDf, proteinName)
        os.remove(noCoPdb)
    # for data with no bound cofactor
    else:
        # use MSMS to split into core and exterior dataframes
        exteriorDf, coreDf = findCoreExterior(pdbFile=pdbFile, pdbDf=pdbDf,
                                            proteinName=proteinName, msmsDir=msmsDir,
                                            outDir=outDir)
        # use fpocket to identify caves 
        caveDfs, pocketTags, fpocketFeatures = gen_multi_cave_regions(outDir=outDir,
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
    for caveDf, pocketTag, fpocketDf in zip(caveDfs, pocketTags, fpocketFeatures):
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
                    extPropertiesDf,corePropertiesDf,cavePropertiesDf, fpocketDf]
        if optionsInfo["genCofactorLabels"]:
            label = labelsDf.loc[:,pocketName]
            label.rename("Cofactor",inplace=True)
            dfsToConcat.append(label)

        ## SORT OUT INDEXES
        for df in dfsToConcat:
            df.index = [pocketName]

        featuresDf = pd.concat(dfsToConcat, axis=1)
        ## GENERATE CATEGORICAL FEATURES
        if optionsInfo["genAminoAcidCategories"]:
            featuresDf = make_amino_acid_category_counts(dataDf = featuresDf,
                                             optionsInfo = optionsInfo)
            
        ## NORMALSIE COUNTS BY REGION SIZE
        if optionsInfo["normaliseCounts"]:
            featuresDf = normalise_counts_by_size(dataDf = featuresDf,
                                                  aminoAcidNames = aminoAcidNames,
                                                  optionsInfo = optionsInfo)


        # Save featuresDf to a CSV file
        saveFile = p.join(outDir, f"{pocketName}_features.csv")
        featuresDf.to_csv(saveFile, index=True)



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
def merge_csv_files(csvFiles, outFile):
    chunk_size = min((10, len(csvFiles)))  # Adjust this based on available memory
    for i in range(0, len(csvFiles), chunk_size):
        chunkFiles = csvFiles[i:i + chunk_size]
        dfs = [pd.read_csv(csvFile, index_col="Unnamed: 0") for csvFile in chunkFiles]
        merged_df = pd.concat(dfs, axis=0)
        merged_df.to_csv(outFile, mode='a', header=not os.path.exists(outFile), index=True)
    remainingFiles = csvFiles[i+chunk_size:]
    if not len(remainingFiles) == 0:
        dfs = [pd.read_csv(csvFile, index_col="Unnamed: 0") for csvFile in remainingFiles]
        merged_df = pd.concat(dfs, axis=0) 
        merged_df.to_csv(outFile, mode='a', header=not os.path.exists(outFile), index=True)

    for csvFile in csvFiles:
        os.remove(csvFile)

###############################################################################################
def batch_list(inputList, nBatches):
    inLength = len(inputList)
    batchSize = inLength // nBatches
    remainder = inLength % nBatches

    batches = [inputList[i * batchSize + min(i, remainder):(i + 1) * batchSize + min(i + 1, remainder)]
                for i in range(nBatches)]
    return batches


########################################################################################
        
main()