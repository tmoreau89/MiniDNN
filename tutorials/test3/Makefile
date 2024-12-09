.PHONY: all
all: test
# This rule tells make how to build hello from iris_model.cpp
test: iris_model.cpp
	g++ -O2 -I../../include -I../../../eigen iris_model.cpp -o iris_model.o

# This rule tells make to delete the program
.PHONY: clean 
clean:
	rm -f iris_model.o


