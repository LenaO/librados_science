import scirados
import numpy as np
print(dir(scirados))
test = scirados.RadosDataSet("TestDataSet", "image_cache")

test2d = scirados.RadosDataSet("TestDataSet2D", "image_cache")

data = np.array([123.42,42,42,3,534.5344, 432], dtype=np.float64)
data2d = np.array([[123.42,42,42,3,534.5344, 432], [1.2342,4.322,4.2,3,32.422, 23]], dtype=np.float64)
data2 = np.array([32.3,3.2], dtype=np.float64)
data22d = np.array([[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]], dtype=np.float64)

#print(data.dtype)
test.writeData(data)

test2d.writeData(data2d)


#test.writeBox(data2, xstart=2)
print(data22d[1:3,0:2])
test2d.writeBox(data22d[1:3,0:2], xstart=2, ystart=0)
#test.writeData("test", data2)
#print("here\n")
test_data = test.readData()
#print(test_data)
print(test2d.readData())
#test_data2 = test.readData("test2")
print(test2d.readBox(xslice=slice(0,2), ystart=1))
print(test2d.getDims())
#print(test.ObjectExists("test2"))
#print(test.writeBox("test2", slice(0,20,2), slice(7,22), data))
