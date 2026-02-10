import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(dropout_rate=0.5, extra_layer=False, use_noise=False):
    """
    Construieste modelul CNN flexibil pentru experimentele de optimizare.
    """
    input_shape = (64, 64, 1)

    model = models.Sequential()
    
    # 1. Preprocesare
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1./255))
    
    # Exp 5: Zgomot Gaussian (doar daca e cerut)
    if use_noise:
        model.add(layers.GaussianNoise(0.1))

    # Augmentare standard (pentru toate)
    model.add(layers.RandomFlip("horizontal_and_vertical"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    # 2. CNN Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 3. Clasificare
    model.add(layers.Flatten())
    
    # Exp 3: Extra Hidden Layer
    if extra_layer:
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    else:
        # Varianta standard
        model.add(layers.Dense(128, activation='relu'))
        
    model.add(layers.Dropout(dropout_rate)) 
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model