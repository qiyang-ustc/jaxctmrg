# Define a variable with all the file names
ALL_FILES = $(wildcard *)

# Define the gdoc target to depend on all files
gdoc: $(ALL_FILES)
	hatch run pdoc jaxctmrg -o ./docs

pytest: $(ALL_FILES)
	hatch run pytest