# Datasets

## EurostatKG

The public repository includes preprocessing code, semantic metadata summaries,
configuration provenance and curated final results. Raw source resources are
not mirrored here.

Example:

```powershell
.\commands\preprocess_dataset.ps1 `
  -Dataset eurostatkg `
  -GraphDir data\raw\eurostatkg\graph `
  -OntologyDir data\raw\eurostatkg\ontology `
  -OutputDir data\processed\eurostatkg
```

## DBpedia

The preprocessing entry point accepts the DBpedia ontology, English instance
types and English mapping-based object triples:

```powershell
.\commands\preprocess_dataset.ps1 `
  -Dataset dbpedia `
  -Ontology data\raw\dbpedia\dbpedia_2016-10.owl `
  -InstanceTypes data\raw\dbpedia\instance_types_en.ttl.bz2 `
  -MappingBasedObjects data\raw\dbpedia\mappingbased_objects_en.ttl.bz2 `
  -OutputDir data\processed\dbpedia
```

The paper evaluation uses a fixed DBpedia query subset. Its identifier file
must be reconstructed or obtained with the artifact release.

## DrugBank

The original DrugBank XML is licensed and is not redistributed. Authorized
users can run:

```powershell
.\commands\preprocess_dataset.ps1 `
  -Dataset drugbank `
  -DrugBankXml data\raw\drugbank\full_database.xml `
  -OutputDir data\processed\drugbank
```

## Excluded artifacts

The repository intentionally excludes raw restricted data, large split files,
entity maps, checkpoints, complete Top-M windows and full query-level outputs.
Curated publication-level summaries remain under `results/frozen/`.
