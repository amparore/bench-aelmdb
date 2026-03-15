# Simple, clean Makefile for building the three test programs
# Usage:
#   make            # build all
#   make slice      # build rbsr_aelmdb_slice_tests
#   make advanced   # build rbsr_aelmdb_advanced_test
#   make clean

CXX      := g++
MKDIR_P  ?= mkdir -p
# CXXFLAGS := -std=c++20 -g -pg
CXXFLAGS := -std=c++20 -O3 -DNDEBUG

# Include / library paths (as in your commands)
INCLUDES := -I../negentropy-aelmdb/cpp/ \
            -I../negentropy-aelmdb/cpp/negentropy/storage/ \
            -I../lmdbxx-aelmdb/include/ \
			-I../aelmdb/ \
			-I. -I/opt/homebrew/include/
LDFLAGS  := -L/opt/homebrew/lib/ #-g -pg
LDLIBS   := -lcrypto

# Prebuilt objects you link against
LINK_OBJS := ../aelmdb/obj/mdb.o ../aelmdb/obj/midl.o
BIN ?= bin

# Programs
PROGS := $(BIN)/rbsr_aelmdb_advanced_test $(BIN)/rbsr_aelmdb_slice_tests

.PHONY: all clean slice advanced
all: $(PROGS)

$(BIN)/rbsr_aelmdb_slice_tests: rbsr_aelmdb_slice_tests.cpp $(LINK_OBJS)
	$(MKDIR_P) bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BIN)/rbsr_aelmdb_advanced_test: rbsr_aelmdb_advanced_test.cpp $(LINK_OBJS)
	$(MKDIR_P) bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LDLIBS) -o $@

# Convenience aliases
slice: $(BIN)/rbsr_aelmdb_slice_tests
advanced: $(BIN)/rbsr_aelmdb_advanced_test

clean:
	$(RM) $(PROGS)