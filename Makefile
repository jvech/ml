CC 		= clang
CFLAGS 	= -std=c11 -Wall -g
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
	./${BIN}

debug: $(BIN)
	gdb $< --tui

clean:
	@rm $(OBJS) $(OBJDIR) -rv
