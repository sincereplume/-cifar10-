from tensorflow import keras
import matplotlib.pyplot as plt
classes = ['plane','car','bird','cat','deer',
          'dog','frog','horse','ship','truck']
def predict(pic_path,models_path):
    model = keras.models.load_model(models_path)
    image = keras.utils.load_img(pic_path,target_size=(32,32))
    image = keras.utils.img_to_array(image)
    image = image.reshape(1,32,32,3)
    image = image.astype('float32')
    image=image/255
    predict_result = model.predict(image)
    res_dict = dict()
    for i in range(10):
        res_dict[predict_result[0][i]] = classes[i]
    probs = predict_result[0]
    probs.sort()
    probs = probs[::-1]
    best_3_probs = probs[:3]
    for i in range(len(best_3_probs)):
        print("{}: {}%".format(res_dict[best_3_probs[i]],(best_3_probs[i]*100).round(2)))
    plt.imshow

path = str(input("please enter the picture path:"))
models = str(input("please enter the model path:"))
predict(path,models)