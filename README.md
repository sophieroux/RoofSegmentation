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
- where the alpha channel is below 128, color is also black. that pixel is missing, so it is left out of the count and drawn as backhround in the mask that is sent
- the gray label fades at roof edge. brighter than 127 is roof, darker is not.


