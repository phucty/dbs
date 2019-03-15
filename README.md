# Demo DBS

The demo for the paper of "Semantic Labeling for Numerical Values: Distribution Base Similarities". This paper aims to provide a new distribution-based approaches which addressed the limitation of the p-value based similarity approaches. 

### License
MIT License

### Contact
Phuc Nguyen - phucnt@nii.ac.jp

The demo conducts on 
- 4 Datasets: City Data [1], Wikidata, DBpedia, and Open Data[2]. 
- 5 evaluation methods: SemanticTyper[1], DSL[2], DBS1 (), DBS2, DBSinf.

### Code structure
- test_labeling.ipynb - python workbook demo
- attribute_obj.py - numerical attribute: organize the semantic label and numerical values
- distances.py - implementation of numerical distance between two bags of numbers (BON)
- io_worker.py - data loader
- labeling.py - implementation of semantic labeling with similarity searching

### Reference
- [1] Ramnandan et al., Assigning semantic labels to data sources. ESWC 2015.
- [2] Pham et al., Semantic labeling: a domain-independent approach. ISWC 2016.
- [3] Phuc Nguyen et al. Semantic Labeling for Numerical Values with Deep Metric Learning, JIST 2018.