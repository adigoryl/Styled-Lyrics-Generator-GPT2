# Conditional Lyrics Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Gl5zGR8vQM6q7fV0XAPvWg5sX1ZNgDOs)

This work utilises a pre-trained GPT2-small model from Transformer library. The pre-trained model was fine-tuned using a lyrics dataset that comprises of 15.5K unique songs that are accompanied by metadata, that is, the genre, artist, year, album and song name. By using an appropiate training input construction and feeding strategy the GPT2 outputs can be constrained with the aformentioned lyrics' features, in order to produce mashup song. For example, one could constrain the lyrics to look like from the "Queen" band, however, in a "Country" genre style. The model can be used as a inspirational tool for creative lyrics composition.

Open an interactive demo of the model in Google Colab (requires a google account): https://colab.research.google.com/drive/1Gl5zGR8vQM6q7fV0XAPvWg5sX1ZNgDOs 

