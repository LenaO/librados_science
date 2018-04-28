import radosObject.radosObject as ro
import numpy as np
dir(ro)
#test_file = ro.RadosObject("test_schema", "lena_testing", "open_or_create")
#test_file["c"]["DataSet"]=np.random.randint(0,1024,size=(32,32),dtype=np.uint32)
#print(test_file["root_set"][:,:])
#print(test_file["c"]["DataSet"][0:10,10:20])
#test_file["c"]["DataSet"][3:5,2:4]=np.zeros((2,2), np.uint32)
#print(test_file["c"]["DataSet"][0:10,0:10])
#
test_file2 = ro.RadosObject("B20_0535_TS01_Pyramid.tif", "lena_testing", "open_or_create")
print (test_file2)
test_file2.delete()

