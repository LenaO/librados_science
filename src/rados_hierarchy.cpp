#include "rados_hierarchy.h"

using namespace JRados;

JRadosHObject::JRadosHObject(std::string name, std::string pool, int mode) :
    _name(name), _pool_name(pool), _schema(name, pool), _data_set(nullptr), _changed(false), _is_leaf(false)
{
    std::string desc;

    switch (mode){
        case OPEN:
            desc = _schema.getSchema();
//            std::cout<<desc<<" "<<name<<std::endl;
            _node = new conduit::Schema(desc);
            break;
        case CREATE:
            if(_schema.exist())
                std::cerr<<"Object "<<name<<"already exist "<<std::endl;
            exit(-1);
            _node = new conduit::Schema();
            _changed = true;
            break;
        case OPEN_OR_CREATE:
            if(_schema.exist()) {
                desc = _schema.getSchema();
                std::cout<<desc<<" "<<name<<std::endl;
                _node = new conduit::Schema(desc);
            }
            else {
                _node =  new conduit::Schema(); 
                _changed = false;
            }
            break;
    }
    _root=this;
    _is_root = true;
}
void remove_empty(conduit::Schema* node){


    int children = node->number_of_children(); 
    for(size_t i = 0; i<children; i++){

        if((*node)[i].dtype().is_empty()){
            node->remove(i);
        }
        else
            remove_empty(&((*node)[i]));

    }
}

int JRadosHObject::save(){
    int ret = 0;
    if(_is_root) {
        if(_changed) {
            remove_empty(_node);
            ret = _schema.writeSchema(*_node);
        }
    }
    return 0;
}
