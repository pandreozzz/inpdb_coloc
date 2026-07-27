FC     = gfortran
FFLAGS = -O2 -fPIC -fopenmp

SRC = src/physics/f_src/fvertintp.f90
LIB = src/physics/f_src/fvertintp.so

$(LIB): $(SRC)
	$(FC) $(FFLAGS) -shared -o $@ $<

clean:
	rm -f $(LIB)
