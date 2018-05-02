import scirados
import conduit
import json
import numpy

class RadosObject:
    def __init__(self, name, pool_name="", mode="create", parent=0, root=0):
        if not pool_name is "":
            self._root = self
            self._radosSchema = scirados.RadosSchema(name, pool_name)
            self._exist = self._radosSchema.exist()
            self._cschema = self._open(mode)
            self._name = name
            self._basename = name
            self._pool_name = pool_name
            self._parent = self
        elif not root is 0 and parent is 0:
            self._root = root
            self._radosSchema = None
            self._cschema = root._cschema[name]
            self._pool_name = root._pool_name 
            self._basename = name
            self._name = root._name+"/"+name 
        elif not root is 0 and not parent is 0:
            self._radosSchema =None
            self._root = root
            self._cschema = parent._cschema[name]
            self._pool_name = parent._pool_name
            self._name = parent._name+"/"+name 
            self._basename = name
            self._parent = parent
        self._dataSet = None
        self._children = dict()


    def delete(self):
        self.save()
        if self._cschema.dtype().is_number():
            if  self._dataSet is None:
                self._dataSet=scirados.RadosDataSet(self._name, self._pool_name)
            self._dataSet.remove()
        else:
            for k in self.keys():
                self[str(k)].delete()
            if self._radosSchema is not None:
                self._radosSchema.remove()

    def _open(self, mode):
        if( mode == "open" or mode == "open_or_create") and self._exist is True:
            s = conduit.Schema(self._radosSchema.getSchema())
        elif mode == "open" and self._exist is False:
            print "Cannot open object"
            return None
        elif (mode == "create" or mode =="open_or_create")  and self._exist is False:
            s = conduit.Schema("{}")
        elif mode is "create" and self._exist is True:
            print("object allready exist")
            return None
        else:
            print("mode not supported")
            return None
        return s

    def __str__(self):
        return str(self._cschema)

    def save(self):
        if self._radosSchema is None:
            self._root.save()
        else:
             self._radosSchema.writeSchema(str(self._cschema))

    def readBox(self, slicex, slicey=None):
        if(not self._cschema.dtype().is_number()):
            print("Not a dataset, return ")
            return None
        if  self._dataSet is None:
            self._dataSet=scirados.RadosDataSet(self._name, self._pool_name)
        if slicey is None:
            data= self._dataSet.readBox(xslice=slicex)
        else:
            data= self._dataSet.readBox(xslice=slicex, yslice=slicey)
        return data

    def writeBox(self,data, slicex, slicey=None):
        if(not self._cschema.dtype().is_number()):
            print("Not a dataset, return ")
            return None
        if  self._dataSet is None:
            self._dataSet=scirados.RadosDataSet(self._name, self._pool_name)
        if slicey is None:
            data= self._dataSet.writeBox(data, xslice=slicex)
        else:
            data= self._dataSet.writeBox(data,xslice=slicex, yslice=slicey)
        return data

    def writeData(self, data):
        new = self._cschema.dtype()
        new.set(data.dtype.name, data.size,0, data.dtype.itemsize, data.dtype.itemsize)
        self._parent._cschema[self._basename] = new        #self._cschema.dtset(conduit.Schema(str(new)))
        if self._dataSet is None:
            self._dataSet = scirados.RadosDataSet(self._name, self._pool_name) 
        self._dataSet.writeData(data)
        self.save()

    def keys(self):
        my_json = json.loads(str(self._cschema))
        return my_json.keys()

    def isDataSet(self):
        return  self._cschema.dtype().is_number()

    def readData(self):
        if(not self._cschema.dtype().is_number()):
            print("Not a dataset, return ")
            return None
        if  self._dataSet is None:
            self._dataSet=scirados.RadosDataSet(self._name, self._pool_name)
        return self._dataSet.readData()

    def  __getitem__(self, index):
            if isinstance(index, basestring):
                index = index.strip("/")
                if index in self._children:
                    return self._children[index]
                else:
                    if(not self._cschema.dtype().is_number()):
                        self._children[index] = RadosObject(index, parent=self, root=self._root) 
                        return self._children[index]
                    else:
                        return None
            elif isinstance(index, tuple):
                if( isinstance(index[0], int) and  isinstance(index[1], int)):
                    return self.readBox(slice(index[0],index[0]+1), slice(index[1], index[1]+1))
                elif( isinstance(index[0], slice) and  isinstance(index[1], slice)):
                    return self.readBox(index[0], index[1])
                elif( isinstance(index[0], int) and  isinstance(index[1], slice)):
                    return self.readBox(slice(index[0],index[0]+1),index[1])
                elif( isinstance(index[0], slice) and  isinstance(index[1], int)):
                    return self.readBox(index[0], slice(index[1], index[1]+1))
            elif isinstance(index,int):
                    return self.readBox(slice(index,index+1))
            elif isinstance(index,slice):
                    return self.readBox(index)
            else:
                return None

    def __setitem__(self, index, item):
        if not isinstance(item , numpy.ndarray):
            print("currently, only numpy types are supported")
            return
        if isinstance(index, basestring):
            index = index.strip("/")
            if index in self._children:
                tmp = self._children[index]
            else:
                if(not self._cschema.dtype().is_number()):
                    self._children[index] = RadosObject(index, parent=self, root=self._root) 
                    tmp = self._children[index]
                else:
                    return None
            tmp.writeData(item)
        elif isinstance(index, tuple):
           if( isinstance(index[0], int) and  isinstance(index[1], int)):
               return self.writeBox(item,slice(index[0],index[0]+1), slice(index[1], index[1]+1))
           elif( isinstance(index[0], slice) and  isinstance(index[1], slice)):
               return self.writeBox(item, index[0], index[1])
           elif( isinstance(index[0], int) and  isinstance(index[1], slice)):
               return self.writeBox(item, slice(index[0],index[0]+1),index[1])
           elif( isinstance(index[0], slice) and  isinstance(index[1], int)):
               return self.writeBox(item,index[0], slice(index[1], index[1]+1))
        elif isinstance(index,int):
            return self.writeBox(item,slice(index,index+1))
        elif isinstance(index,slice):
            return self.writeBox(item,index)

