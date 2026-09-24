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
