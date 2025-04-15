# Brief overview of Finetuning methodology followed

All the 8 trained models were first tested on the new design. The best performing (un-finetuned) model was then picked and finetuned on the new dataset for 50 epochs with 0.0005 learning rate and a patience of 10.

The trained models were finetuned on the `sin` design for comparision. This folder contains the experimental results of the finetuning.

Commands used:
```bash
cd finetuning
python clusterRecipes.py // You can update the dataset location and initial and count at top of the file
cd ..
python finetuning/finetune.py sin 5416 // sin is the design_name and 5416 is the initial area for normalisation
// Note that cd .. and then running from outside is necessary
```