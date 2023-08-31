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
	@./${BIN} train data/sample_data.json | tee data/train_history.txt
	@./${BIN} predict data/sample_data.json | jq -r '.[] | [values[] as $$val | $$val] | @tsv' > ./data/net_data.tsv
	@gnuplot -p utils/plot.gpi

debug: build
	gdb -x utils/commands.gdb --tui --args ${BIN} train -a 230 data/sample_data.json -e 150
	gdb -x utils/commands.gdb --tui --args ${BIN} predict data/sample_data.json

clean:
	@rm $(OBJS) $(OBJDIR) -rv
