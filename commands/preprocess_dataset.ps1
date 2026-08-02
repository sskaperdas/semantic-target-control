[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("dbpedia", "drugbank", "eurostatkg")]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [string]$Ontology,
    [string]$InstanceTypes,
    [string]$MappingBasedObjects,
    [string]$DrugBankXml,
    [string]$GraphDir,
    [string]$OntologyDir,

    [int]$Seed = 42,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

switch ($Dataset) {
    "dbpedia" {
        foreach ($Value in @($Ontology, $InstanceTypes, $MappingBasedObjects)) {
            if ([string]::IsNullOrWhiteSpace($Value)) {
                throw "DBpedia requires -Ontology, -InstanceTypes and -MappingBasedObjects."
            }
        }

        & $Python `
            "$Root\scripts\preprocessing\preprocess_dbpedia.py" `
            --ontology $Ontology `
            --instance-types $InstanceTypes `
            --mappingbased-objects $MappingBasedObjects `
            --output-dir $OutputDir `
            --seed $Seed `
            --deduplicate-triples
    }

    "drugbank" {
        if ([string]::IsNullOrWhiteSpace($DrugBankXml)) {
            throw "DrugBank requires -DrugBankXml."
        }

        & $Python `
            "$Root\scripts\preprocessing\preprocess_drugbank_xml.py" `
            --input $DrugBankXml `
            --output-dir $OutputDir `
            --dataset-name DrugBank `
            --seed $Seed
    }

    "eurostatkg" {
        foreach ($Value in @($GraphDir, $OntologyDir)) {
            if ([string]::IsNullOrWhiteSpace($Value)) {
                throw "EurostatKG requires -GraphDir and -OntologyDir."
            }
        }

        & $Python `
            "$Root\scripts\preprocessing\preprocess_eurostatkg.py" `
            --dataset-name EurostatKG `
            --graph-dir $GraphDir `
            --ontology-dir $OntologyDir `
            --output-dir $OutputDir `
            --seed $Seed
    }
}

if ($LASTEXITCODE -ne 0) {
    throw "Preprocessing failed for $Dataset."
}
