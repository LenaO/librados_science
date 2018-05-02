import scirados
import numpy as np
print(dir(scirados.RadosCache))
print(dir(scirados))

test = scirados.RadosCache(name="image_cache")
data = np.array([123.42,42,42,3,534.5344, 432], dtype=np.float32)
data2 = np.array([123,42,42, 432], dtype=np.uint8)
print(data.dtype)
test.writeData("test2", data)
test.writeData("test", data2)
print("here\n")
test_data = test.readData("test")
print(test_data)
test_data2 = test.readData("test2")
print(test_data2)
print(test.ObjectExists("test2"))
