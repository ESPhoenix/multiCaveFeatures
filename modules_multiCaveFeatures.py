import os
from os import path as p
import numpy as np
import pandas as pd
import subprocess
from shutil import copy, rmtree
from pdbUtils import *
from scipy.spatial.distance import cdist
import yaml
###########################################################################################################
def normalise_counts_by_size(dataDf, aminoAcidNames, optionsInfo):
    elements = ["C","N","O","S"]
    for region in ["cave","core","ext"]:
        ## get totl number of AAs in region
        totalCounts = dataDf[f"{region}.total"]
        # create a list of features to be normalised
        featureList = []
        if optionsInfo["keepIndividualCounts"]:
            for aminoAcid in aminoAcidNames:
                featureList.append(f"{region}.{aminoAcid}")
        if optionsInfo["genAminoAcidCategories"]:
            for category in ["hydrophobic", "aromatic","polar_uncharged","cationic","anionic"]:
                featureList.append(f"{region}.{category}")
        for element in elements:
            featureList.append(f"{region}.{element}")
        ## divide by number of AAs in region
        dataDf[featureList] = dataDf[featureList].div(totalCounts,axis=0)
        ## remove total count features if specified
        if not optionsInfo["keepTotalCounts"]:
            dataDf.drop(columns = [f"{region}.total"], inplace=True)
    return dataDf
###########################################################################################################
def make_amino_acid_category_counts(dataDf, optionsInfo):
    hydrophobicAAs = ["ALA","VAL","ILE","LEU","MET","GLY","PRO"]
    aromaticAAs = ["PHE","TYR","TRP"]
    polarUncharged = ["SER", "THR", "ASN","GLN","HIS","CYS"]
    cationicAAs = ["ARG","LYS"]
    anionicAAs = ["ASP","GLU"]
    aaCategories = {"hydrophobic": hydrophobicAAs,
                    "aromatic": aromaticAAs,
                    "polar_uncharged": polarUncharged,
                    "cationic": cationicAAs,
                    "anionic": anionicAAs}
    
    for region in ["cave","core","ext"]:
        for category in aaCategories:
            colNames = [f"{region}.{AA}" for AA in aaCategories[category]]
            dataDf.loc[:,f"{region}.{category}"] = dataDf[colNames].sum(axis=1)
            if not optionsInfo["keepIndividualCounts"]:
                dataDf.drop(columns = colNames, inplace = True)
                
    return dataDf
###########################################################################################################
def vert2df(vertFile):
    x = []
    y = []
    z = []
    with open(vertFile,"r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) == 4:
                continue
            x.append(cols[0])
            y.append(cols[1])
            z.append(cols[2])
    data = {"X" : x, "Y" : y, "Z" : z}
    pdbDf = pd.DataFrame(data)
###########################################################################################################
def area2df(areaFile):
    ses =[]
    with open(areaFile,"r") as file:
        for line in file:
            if "Atom" in line:
                continue
            cols = line.split()
            ses.append(float(cols[1]))
    data = {"SES":ses}
    pdbDf = pd.DataFrame(data)
    return pdbDf
###########################################################################################################
def initialiseAminoAcidInformation(aminoAcidTable):
    AminoAcidNames = ["ALA","ARG","ASN","ASP","CYS",
                      "GLN","GLU","GLY","HIS","ILE",
                      "LEU","LYS","MET","PHE","PRO",
                      "SER","THR","TRP","TYR","VAL"]  

    # read file with amino acids features
    aminoAcidProperties = pd.read_csv(
        aminoAcidTable, sep="\t", index_col=1
    )
    aminoAcidProperties.index = [el.upper() for el in aminoAcidProperties.index]
    aminoAcidProperties = aminoAcidProperties.iloc[:, 1:]

    return AminoAcidNames, aminoAcidProperties

########################################################################################
def getPdbList(dir):
    pdbList=[]
    idList=[]
    for file in os.listdir(dir):
        fileData = p.splitext(file)
        if fileData[1] == '.pdb':
            idList.append(fileData[0])
            pdbList.append(p.join(dir,file))
    return idList, pdbList

########################################################################################
def findCoreExterior(pdbFile,msmsDir,pdbDf,proteinName,outDir):
    # change working directory so MSMS can find all the files it needs
    os.chdir(msmsDir)
    # find executables
    pdb2xyzrExe = "./pdb_to_xyzr"
    msmsExe = "./msms.x86_64Linux2.2.6.1"
    # convert pdb file to MSMS xyzr file 
    xyzrFile = p.join(outDir, f'{proteinName}.xyzr')
    command = f"{pdb2xyzrExe} {pdbFile} > {xyzrFile}"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # use MSMS to create an area file
    areaOut = p.join(outDir,proteinName)
    command = f"{msmsExe} -if {xyzrFile} -af {areaOut}"
    subprocess.run(command, shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    areaFile=p.join(outDir,f"{proteinName}.area")
    # convert area file to dataframe, merge with main pdb dataframe
    areaDf = area2df(areaFile=areaFile)
    pdbDf = pd.concat([pdbDf,areaDf],axis=1)

    # Group by residue and calculate the average SES score
    meanSesPerResidue = pdbDf.groupby('RES_ID')['SES'].mean()

    # Get residue sequences with average SES > 1
    exteriorResiduesIndex = meanSesPerResidue[meanSesPerResidue > 1].index

    # Split the DataFrame based on average SES > 1
    exteriorDf = pdbDf[pdbDf['RES_ID'].isin(exteriorResiduesIndex)]
    coreDf = pdbDf[~pdbDf['RES_ID'].isin(exteriorResiduesIndex)]

    # clean up
    os.remove(xyzrFile)
    os.remove(areaFile)

    return exteriorDf, coreDf
########################################################################################
def element_count_in_region(regionDf,regionName,proteinName):
    ## INITIALISE ELEMENT COUNT DATAFRAME ##
    columnNames=[]
    for element in ["C","N","O","S"]:
        columnNames.append(f"{regionName}.{element}")
    elementCountDf = pd.DataFrame(columns=columnNames,index=[proteinName])
    ## COUNT ELEMENTS IN REGION, RETURN ZERO IF REGION HAS NONE OR DOES NOT EXIST
    for element in ["C","N","O","S"]:
        try:
            elementCountDf.loc[:,f'{regionName}.{element}'] = regionDf["ELEMENT"].value_counts()[element]
        except:
            elementCountDf.loc[:, f'{regionName}.{element}'] = 0

    return elementCountDf
########################################################################################
def amino_acid_count_in_region(regionDf, regionName, proteinName, aminoAcidNames):
    ## INITIALSE AMINO ACID COUNT DATAFRAME ##
    columnNames=[]
    for aminoAcid in aminoAcidNames:
        columnNames.append(f"{regionName}.{aminoAcid}")
    aaCountDf = pd.DataFrame(columns=columnNames,index=[proteinName])

    ## GET UNIQUE RESIDUES ONLY ##
    uniqueResiduesDf = regionDf.drop_duplicates(subset = ["RES_ID"])

    ## COUNT AMINO ACIDS IN REGION, RETURN ZERO IF REGION HAS NONE OR DOES NOT EXIST
    totalResidueCount = []
    for aminoAcid in aminoAcidNames:
        try: 
            aaCountDf.loc[:,f'{regionName}.{aminoAcid}'] = uniqueResiduesDf["RES_NAME"].value_counts()[aminoAcid]
        except:
            aaCountDf.loc[:,f'{regionName}.{aminoAcid}'] = 0

    aaCountDf.loc[:,f"{regionName}.total"] = aaCountDf.sum(axis=1)
    return aaCountDf
########################################################################################
def calculate_amino_acid_properties_in_region(aaCountDf, aminoAcidNames, aminoAcidProperties, proteinName, regionName):
    ## INITIALISE PROPERTIES DATAFRAME ##
    columnNames = []
    for property in aminoAcidProperties.columns:
        columnNames.append(f"{regionName}.{property}")
    propertiesDf = pd.DataFrame(columns=columnNames, index=[proteinName])
    
    ## LOOP THROUGH PROPERTIES SUPPLIED IN AMINO_ACID_TABLE.txt
    for property in aminoAcidProperties:
        propertyValue=0
        for aminoAcid in aminoAcidNames:
            try:
                aaCount = aaCountDf.at[proteinName,f"{regionName}.{aminoAcid}"]
            except KeyError:
                aaCount = 0
            aaPropertyvalue = aminoAcidProperties.at[aminoAcid,property]
            value = aaCount * aaPropertyvalue
            propertyValue += value 
        try:
            totalAminoAcids=aaCountDf.at[proteinName,f'{regionName}.total']
        except KeyError:
            totalAminoAcids=0
        if not totalAminoAcids == 0:
            propertyValue = propertyValue / totalAminoAcids
        propertiesDf[f'{regionName}.{property}'] = propertyValue

    return propertiesDf

########################################################################################
def gen_multi_cave_regions(outDir,pdbFile):
    proteinName = p.splitext(p.basename(pdbFile))[0]

    pocketDir = p.join(outDir,proteinName)
    os.makedirs(pocketDir,exist_ok=True)
    pocketPdb = p.join(pocketDir,f"{proteinName}.pdb")
    copy(pdbFile,pocketPdb)

    os.chdir(pocketDir)
    ## Run FPocket
    minSphereSize = "3.4"
    maxSphereSize = "10"

    subprocess.call(["fpocket","-f",pocketPdb,"-m",minSphereSize,"-M",maxSphereSize],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Read FPocket results into a dictionary, via  yaml file
    fpocketOutDir = p.join(pocketDir,f"{proteinName}_out")
    fpocketPdbDir = p.join(fpocketOutDir, "pockets")
    fpocketInfo = p.join(fpocketOutDir,f"{proteinName}_info.txt")
    if not p.isfile(fpocketInfo):
        print(proteinName)
        return None, None
    info = fpocket_info_to_dict(fpocketInfo,fpocketOutDir)
    # generate unique identifiers for pockets
    pocketDfs = []
    pocketTags = []
    for pocketId in info:
        pocketNumber = pocketId.split()[1]
        pocketPdb = p.join(fpocketPdbDir, f"pocket{pocketNumber}_atm.pdb")
        pocketVert = p.join(fpocketPdbDir, f"pocket{pocketNumber}_vert.pqr")
        if info[pocketId]["Volume"] < 1000:
            os.remove(pocketPdb)
            os.remove(pocketVert)
            continue
        pocketPdb = p.join(fpocketPdbDir, f"pocket{pocketNumber}_atm.pdb")
        pocketDf = pdb2df(pocketPdb)
        pocketDfs.append(pocketDf)
        pocketTag = ("_").join(pocketId.split())
        pocketTags.append(pocketId)
        os.remove(pocketVert)

    # use fpocket output as features
    fpocketFeatures = []
    for pocketKey in info:
        fpocketFeatureDict = {}
        for key in info[pocketKey]:
            newKey = "cave."+"_".join(key.split())
            fpocketFeatureDict[newKey] = info[pocketKey][key]
        tmpDf = pd.DataFrame(fpocketFeatureDict, index=[1])
        fpocketFeatures.append(tmpDf)
    ## CLEAN UP POCKET DIR ##
    rmtree(pocketDir)
    return pocketDfs, pocketTags, fpocketFeatures

########################################################################################
def fpocket_info_to_dict(infoFile, fpocketOutDir):
    with open(infoFile,"r") as  txtFile:
        txt = txtFile.read()
    yamlData = txt.replace("\t", " "*2)
    with open(p.join(fpocketOutDir,"info.yaml"),"w") as yamlFile:
        yamlFile.write(yamlData)

    with open(p.join(fpocketOutDir,"info.yaml"),"r") as yamlFile:
        info = yaml.safe_load(yamlFile) 

    return info
########################################################################################
def gen_pocket_tag(df, pocketTags):
    pocketCenter = [df["X"].mean(), df["X"].mean(), df["X"].mean()]
    df.loc[:,"pocketCenter"] = np.linalg.norm(df[["X", "Y", "Z"]].values - np.array(pocketCenter), axis=1)
    df.sort_values(by="pocketCenter", ascending= False, inplace=True)
    for i in range(0,len(df)):
        chain = df.loc[i,"CHAIN_ID"]
        resName = df.loc[i,"RES_NAME"]
        resId = str(df.loc[i,"RES_ID"])
        pocketTag = ":".join([chain,resName,resId])
        if not pocketTag in pocketTags:
            return pocketTag
########################################################################################
def generate_cofactor_labels(caveDfs, pocketTags, cofactorDf, proteinName):
    indexNames = [f"{proteinName}_{pocketTag}" for pocketTag in pocketTags]
    labelsDf = pd.DataFrame(index = indexNames, columns = ["Cofactor"]).fillna("Vacant")
    uniqueCofactorNames = cofactorDf["RES_NAME"].unique()
    for coName in uniqueCofactorNames:
        uniqueCoDf = cofactorDf[cofactorDf["RES_NAME"] == coName]
        minDistances = []
        for caveDf, tag in zip(caveDfs, indexNames):
            # find min distance between caveDf and cofactorDf
            caveCoords = caveDf[["X","Y","Z"]].values
            coCoords = uniqueCoDf[["X","Y","Z"]].values
            distances = cdist(caveCoords, coCoords)
            minDist = distances.min()
            tmpDf = pd.DataFrame({tag:minDist}, index = [coName])
            minDistances.append(tmpDf)
        minDistDf = pd.concat(minDistances, axis=1).T
        minIndex = minDistDf.idxmin(axis=0).to_list()
        minDist = minDistDf.min()
        try:
            if minDist.values < 3.4:
                labelsDf.loc[minIndex,"Cofactor"] = coName
        except:
            print(minDist)
            continue
    return labelsDf.T     


