#include <stdexcept>
#pragma once

template<typename T, int MAX_SIZE>
class fast_static_vector {
public:
    fast_static_vector() : 
        size_(0) 
    { 

    }

    fast_static_vector(int size) :
        size_(size)
    { 
        for(int i = 0; i < size; ++i) { 
            data_[i] = 0; // fixme should be init type
        }

    }

    fast_static_vector(int size, const T init_val) :
        size_(size)
    { 
        for(int i = 0; i < size; ++i) { 
            data_[i] = init_val; 

        }
    }

    T & operator[](int pos){  

        return data_[pos]; 
    }

    T operator[](int pos) const{  

        return data_[pos]; 
    }

    int size() { 
        return size_; 
    }

    bool operator <(const fast_static_vector<T, MAX_SIZE> & b) const { 
        if(size_ == b.size_) { 
            for(int i = 0; i < size_; ++i) { 
                if (data_[i] < b.data_[i]) { 
                    return false; 
                } else if (data_[i] > b.data_[i]) { 
                    return true; 
                }
            }
            return true; 


        } else { 
            std::cout << "size_=" << size_ << " b.size_=" << b.size_ 
                      << std::endl; 
            throw std::runtime_error("can't compare unequal size"); 
        }

    }


private:
    int size_; 
    T data_[MAX_SIZE]; 

}; 
