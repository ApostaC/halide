CXX=clang++
CC=clang
FLAGS= -g -lpthread -lHalide -std=c++14

TARGETS=$(patsubst %.cpp, % ,$(wildcard *.cpp)) 

all: ${TARGETS}
	@echo ${TARGETS}

%: %.cpp
	${CXX} $^ -o $@ ${FLAGS}


