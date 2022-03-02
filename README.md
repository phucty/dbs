# Demo DBS

The demo for the paper of:

P. Nguyen and H. Takeda: Semantic Labeling for Numerical Values: Distribution Base Similarities. Special Inerest Group for Semantic Web and Ontology, Vol. 47, No. 12, 2019
[Link](https://jsai.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=9812&item_no=1&page_id=13&block_id=23)

This paper aims to provide a new distribution-based approaches which addressed the limitation of the p-value based similarity approaches.

### How to run the code
```
git clone https://github.com/phucty/dbs.git
cd dbs
conda create -n dbs python=3.6
conda activate dbs
pip install -r requirements.txt
```
Then you can run the example in [labeling.py](labeling.py)
```
python labeling.py
```
or run the python notebook in [test_labeling.ipynb](test_labeling.ipynb)

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
