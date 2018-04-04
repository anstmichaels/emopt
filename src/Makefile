UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
# Linux
	CC = g++
	SHARED_FLAG = -shared
else
	ifeq ($(UNAME), Darwin)
# OS X
		CC = g++
		SHARED_FLAG = -dynamiclib
	else
		CC = g++
	endif
endif

CXXFLAGS = -Wall -fopenmp -O3 -march=native -DNDEBUG -std=c++11

INCLUDES = -I${EIGEN_DIR} -I${BOOST_DIR}
LDFLAGS = -lpthread

# consider using -lopenblas if large problems are to be solved
grid:
	${CC} -c -fPIC Grid.cpp ${CXXFLAGS} -o Grid.o ${INCLUDES}
	${CC} -c -fPIC Grid_ctypes.cpp ${CXXFLAGS} -o Grid_ctypes.o ${INCLUDES}
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o Grid.so Grid.o Grid_ctypes.o ${LDFLAGS}
	cp Grid.so ../emopt

clean:
	rm -f ./*.o
	rm -f ./*.so
	rm -f ../emopt/*.so
	rm -f ../emopt/*.pyc

