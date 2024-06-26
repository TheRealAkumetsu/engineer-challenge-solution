# engineer-challenge-solution

To use this please add the weights_final.h5 file to this folder, navigate inside and then run (on macOS)

`docker build --platform linux/arm64 -t roberta-sentiment-arm64 .`

which takes about 3 minutes. Afterwards you can run the container via

`docker run --platform linux/arm64 -p 5001:5000 roberta-sentiment-arm64`

and test that it is up via

`curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{"text": "This is a test text", "sentiment": "neutral"}'`

