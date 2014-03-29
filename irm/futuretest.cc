#include <future>
#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>


namespace bt = boost::posix_time; 


double foo(size_t start, size_t end) { 
    double result = 0;
    for (size_t i = start; i < end; ++i)   { 
        result += i; 
    }
    return result; 
}

int main()
{                                               
    double WORK_SIZE = 2e9; 
    std::cout << "HEllo world" << std::endl; 
    float single_time = 0.0;
    for (int THREAD_N = 1; THREAD_N < 10; ++THREAD_N) { 
        std::vector<std::future<double> > results; 
        bt::ptime mst1 = bt::microsec_clock::local_time();

        for(size_t i = 0; i < THREAD_N; ++i) { 
            results.push_back(std::async(std::launch::async, 
                                         foo, i*WORK_SIZE, 
                                         (i+1)*WORK_SIZE)); 
        }
        // get the results
        for(int i = 0; i < THREAD_N; ++i) { 
            results[i].get(); 
        }


        bt::ptime mst2 = bt::microsec_clock::local_time();
        bt::time_duration msdiff = mst2 - mst1;

        if(THREAD_N == 1) { 
            single_time = msdiff.total_milliseconds() ; 
        }

        std::cout << msdiff.total_milliseconds() << " ms "
                  << msdiff.total_milliseconds() / single_time << std::endl;
    }
}
