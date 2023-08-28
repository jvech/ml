CC 		= clang
CFLAGS 	= -std=gnu11 -Wall -g
BIN 	= ml
OBJDIR 	= objs
SRC 	= $(wildcard src/*.c)
HEADERS = $(wildcard src/*.h)
OBJS 	= $(SRC:src/%.c=${OBJDIR}/%.o) 
DLIBS 	= -lm $(shell pkg-config --libs-only-l blas json-c)
.PHONY: clean all run

all: build

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir ${OBJDIR}

$(OBJDIR)/%.o: src/%.c $(HEADERS)
	${CC} -c -o $@ $< ${CFLAGS}

build: $(OBJS)
	${CC} ${DLIBS} -o ${BIN} ${OBJS}

run: build
	./${BIN} train -a 1e-6 data/sample_data.json -e 150

debug: build
	gdb -x utils/commands.gdb --tui --args ${BIN} train -a 230 data/sample_data.json -e 150

clean:
	@rm $(OBJS) $(OBJDIR) -rv
