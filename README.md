# RoofSegmentation: roof masks as a hold-out likelihood ratio 

- 30 aerial images of houses
- 25 of those come with drawing of roof
- task is to draw the roofs of the remaining 5
  
- roof is only a small part of each photo (roofs are red, grey, brown, or nearly black, so a color rule does not work)
- drawings are slightly soft at the edges
- each pixel is turned into a yes or no: brighter than 127 counts as roof
- black rectangles where the satellite tile is missing are left out (not roofs but missing data) 

- model is a small U-Net
- for every pixel it outputs a probability that the pixel is roof
- with only 25 labeled photos, each photo is also flipped and rotated quarter turns during training (I don't invent new houses, but show the same photo again in another orientation, because 25 labeled photos is a small training set) 
- roof drawing has to turn with the photo, or the label would no longer sit on the roof


### Understanding the data

- photos are 256 by 256 PNG files in `data/images`, labels in `data/labels`, same id in the filename. Those folders are not in the repo. `python -m roofseg.download` fetches the zip
- images are RGBA. The label is grayscale.
- The five without a label are the following: 535, 537, 539, 551, and 553
- where the alpha channel is below 128, color is also black. that pixel is missing, so it is left out of the count and drawn as background in the mask that is sent
- the gray label fades at roof edge. brighter than 127 is roof, darker is not.


### What the network has to beat 

- on the training images, around 15% of pixels that are there are roof. This rate I call $p_0$. The null guess assigns every pixel the rate $p_0$.
- five labeled images are left out of the first fit (picked by id, seed 41): 274, 278, 300, 320, 532. the five unlabeled images are never used for training
- for each left out image, the real roof and background pixels are scored under $p_0$ and under the network. the network has to come out ahead. The score is saved from `selection.pt`
- Soft Dice is only in the training loss, because the roof is a small part of the photo.
- the five masks to send come from a second fit, `final.pt`. this run uses all 25 labels, for the same number of epochs the first run has, but starting from new weights. Those masks were not scored.

### What came out 

- the first fit stopped at epoch 15. After that, the five photos the network had not seen got worse
- on four of those five photos, the network's probabilities fit the real roof better than painting 15% on every pixel. On image 278 they fit worse
- image 274: the probabilities are a bit better than 15%, but no pixel reaches 50%. A pixel is painted as roof only when the network is at least half sure, so the drawing is blank
- image 278: a few pixels land on the real roof, and some other pixels are nearly certain and wrong. those certain mistakes make the whole image much worse than the 15% guess
- images 300, 320, and 532: the painted roofs sit on the real roofs. image 320 matches best

  
### How to run

- `make env` builds the conda environment from `environment.yml`. then `conda activate roofseg`
- `python -m roofseg.download` fetches the photos into `data/`
- `python -m roofseg.train` does both fits. it writes `outputs/checkpoints/selection.pt`, `outputs/checkpoints/final.pt`, the five masks in `outputs/predictions/`, and `outputs/metrics.json`
- `python -m roofseg.predict` draws those five masks again from `final.pt`, without training
- `make test` checks that missing tile stays out of the count, and that the roof drawing stays on the roof even when flipped
- `roof_segmentation.ipynb` shows a few images, the table for the left-out ones and the five masks. Run the fit first or else the notebook has nothing to read 




