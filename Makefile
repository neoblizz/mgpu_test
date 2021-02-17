include Makefile.inc

all: test color

test : test.cu $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o bin/test test.cu $(SOURCE) $(ARCH) $(INC)

color : color.cu $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o bin/color color.cu $(SOURCE) $(ARCH) $(INC)