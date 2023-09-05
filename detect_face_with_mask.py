# importe a biblioteca opencv
import cv2

import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")

# defina um objeto de captura de vídeo
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture o vídeo quadro a quadro
    ret, frame = vid.read()

    prediction = model.predict(frame)

    #redimensione a imagem
    img = cv2.resize(frame,(224,224))

    #converta a imagem em um array numpy e aumente a dimensão
    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)

    #normalize a imagem
    normalised_image = test_image/255.0

    #preveja o resultado
    prediction = model.predict(normalised_image)

    print("Previsão: ", prediction)
  
    # Exiba o quadro resultante
    cv2.imshow('quadro', frame)
      
    # Saia da tela com a barra de espaço
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# Após o loop, libere o objeto capturado
vid.release()

# Destrua todas as janelas
cv2.destroyAllWindows()
