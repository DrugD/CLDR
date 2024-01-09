***This is the code for "CLDR: Contrastive Learning Drug Response Models from Natural Language Supervision".***

---


<h2>1.Preparation for work</h2>

  Download the GDSCv2 from [https://www.cancerrxgene.org/](https://www.cancerrxgene.org/downloads/anova).

      Select: 'Genetic Features'
      Screening Set: GDSC2
      Select tissue type: Pan-Cancer
      Select mutation type: Copy number alteration
      
  Then just click the 'Download' button.

Data preprocessing refer to https://github.com/DrugD/TransEDRP.

<h2>2.How to run</h2>

<h3>For pre-train and then fine-tune</h3> 

    python main_zs_CLIP.py
    python main_zs_CLIP_then_MSE.py

<h3>For train just with MSE</h3> 
    
    python main_zs_MSE.py


