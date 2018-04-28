import scirados
import numpy as np
print(dir(scirados.RadosSchema))
test = scirados.RadosSchema("test_schema", "lena_testing")
print(test.writeSchema(test.getSchema()))
