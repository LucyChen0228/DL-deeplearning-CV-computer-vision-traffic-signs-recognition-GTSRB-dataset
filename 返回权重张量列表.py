from keras.models import load_model
model = load_model('traffic_signs.h5')

for layer in model.layers:
    for weight in layer.weights:
        print(weight.name,weight.shape)
