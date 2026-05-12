from tensorflow.keras.models import load_model #type: ignore
import numpy as np
def pred(x_test_):
	model=load_model("handwritten_nb.keras")
	y_hat=model.predict(x_test_)
	return np.argmax(y_hat)