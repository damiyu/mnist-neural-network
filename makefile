softmax:
	python ./src/main.py --experiment test_softmax

gradient:
	python ./src/main.py --experiment test_gradients

momentum:
	python ./src/main.py --experiment test_momentum

regular1:
	python ./src/main.py --experiment test_regularization_1

regular2:
	python ./src/main.py --experiment test_regularization_2

regular3:
	python ./src/main.py --experiment test_regularization_3

relu:
	python ./src/main.py --experiment test_relu

sigmoid:
	python ./src/main.py --experiment test_sigmoid