# Neural Style Transfer with AdaIN

A Flask web app that combines a content image with a style reference using Adaptive Instance Normalization (AdaIN).

## Run locally

```bash
cd NST_Code
pip install -r ../requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser, then upload a content image and a style image.

The app includes the trained model weights in `NST_Code/experiment/final_exp/decoder_final.pth` and `NST_Code/vgg_normalised.pth`.